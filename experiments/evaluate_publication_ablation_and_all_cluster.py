"""Publication-extension ablations and simultaneous all-cluster evaluation.

This module does not alter cluster-head selection or routing.  It applies one
allocator independently to every scheduled cluster in the same network round,
then advances the shared physical environment once with the combined action.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.budget_projection import projection_optimality_diagnostic
from agents.ch_depletion_risk import validate_ch_risk_config
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from envs.step3_lifetime_env import configure_step3_risk
from envs.step3_policy_observation import (
    CH_CONTEXT_NAMES,
    STEP3_CH_CONTEXT_SCHEMA,
    build_step3_observation,
)
from envs.step3_v3_env import episode_service_fairness
from experiments.diagnose_step3_delivery_feasibility import load_agent
from experiments.evaluate_step4_publication_evidence import (
    ScenarioRoleSeparatedScheduledMACEnv,
    independent_schedule_bundle,
    load_trace,
)
from experiments.run_phase3_pilot import build_assets


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path: str | Path) -> str:
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def _load_json(path: str | Path) -> dict:
    return json.loads(resolve(path).read_text(encoding="utf-8"))


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "publication_extension_protocol_v1":
        raise RuntimeError("publication-extension protocol has unexpected status")
    for field in ("source_checkpoint", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact checksum mismatch: {field}")
    trace = contract["external_trace"]
    if sha256(trace["path"]) != trace["sha256"]:
        raise RuntimeError("external trace checksum mismatch")
    training = set(map(int, contract["training_environment_seeds"]))
    pilot = set(map(int, contract["pilot_seeds"]))
    evaluation = set(map(int, contract["evaluation_seeds"]))
    prior = set(map(int, contract["prior_used_seeds"]))
    if training & evaluation or pilot & evaluation or evaluation & prior or pilot & prior:
        raise RuntimeError("training/evaluation seed cohorts are not independent")
    if int(contract["workers"]) * int(contract["threads_per_worker"]) > 16:
        raise RuntimeError("CPU worker contract exceeds 16 logical threads")
    if set(contract["policies"]) != {
        "hta_mac", "hta_no_ch_context", "hta_no_set_context",
        "energy_proportional", "online_primal_dual",
    }:
        raise RuntimeError("publication policy set changed")
    return contract


class ClusterStatistics:
    """Per-rank online state for observations, fairness, and dual control."""

    def __init__(self, nodes: int, qos: Step3QoSConstraintConfig, dual_config: dict):
        self.offered = np.zeros(nodes, dtype=np.int64)
        self.delivered = np.zeros(nodes, dtype=np.int64)
        self.stale = 0
        self.step3_demand_ewma = 0.0
        self.step3_qos_lifetime_preference = 0.5
        self.qos = qos
        initial = dual_config["initial_multiplier"]
        self.dual = {key: float(initial[key]) for key in ("delivery", "stale", "fairness")}
        self.dual_config = dual_config

    @property
    def fairness(self) -> float:
        return episode_service_fairness(self.delivered, self.offered)

    def update_duals(self) -> None:
        demand = max(1, int(self.offered.sum()))
        residuals = {
            "delivery": self.qos.minimum_delivery_ratio - int(self.delivered.sum()) / demand,
            "stale": self.stale / demand - self.qos.maximum_stale_drop_ratio,
            "fairness": self.qos.minimum_queue_fairness - self.fairness,
        }
        eta = float(self.dual_config["learning_rate"])
        maximum = float(self.dual_config["maximum_multiplier"])
        for key, residual in residuals.items():
            self.dual[key] = float(np.clip(self.dual[key] + eta * residual, 0.0, maximum))


def build_environment(contract: dict, scenario: dict, seed: int, horizon: int):
    _, solar, thermal, radio, base_cfg, _ = build_assets(horizon)
    risk = validate_ch_risk_config(_load_json(contract["risk_config"]))
    configure_step3_risk(risk)
    nodes = int(scenario["nodes"])
    field_scale = float(scenario["field_scale"])
    field_size = 100.0 * field_scale
    bs_position = (50.0 * field_scale, 175.0 * field_scale)
    cfg = replace(
        base_cfg,
        max_rounds=horizon,
        initial_energy_j=base_cfg.initial_energy_j * float(scenario["initial_energy_multiplier"]),
        bs_position_m=bs_position,
    )
    trace = (
        load_trace(resolve(contract["external_trace"]["path"]))
        if scenario.get("use_external_trace") else None
    )
    bundle = independent_schedule_bundle(
        seed=int(seed), nodes=nodes, horizon=horizon, field_size=field_size,
        bs_position=bs_position, solar_states=len(solar.initial),
        thermal_states=len(thermal.initial),
    )
    env = ScenarioRoleSeparatedScheduledMACEnv(
        cfg, radio, solar, thermal, idle_energy_enabled=True,
        arrival_rate=float(scenario["arrival_rate"]),
        harvest_multiplier=float(scenario["harvest_multiplier"]),
        trace_w_m2=trace,
    )
    env.reset(seed=int(seed), frozen_snapshot=bundle)
    return env, bundle, risk


def cluster_members(base, cluster: int) -> tuple[int, np.ndarray, np.ndarray]:
    ch = int(base.cluster_heads[cluster])
    members = np.flatnonzero(
        (base.cluster_of == cluster) & (np.arange(base.n_nodes) != ch)
    )
    mask = np.zeros(base.n_nodes, dtype=bool)
    if base.alive[ch]:
        mask[members] = base.alive[members]
    return ch, members, mask


def energy_proportional_cluster_action(base, mask, caps, budget: int, exponent: float):
    action = np.zeros(base.n_nodes, dtype=np.int64)
    score = np.power(
        np.clip(base.energy / max(base.cfg.initial_energy_j, 1e-12), 0.0, 1.0),
        float(exponent),
    )
    for _ in range(int(budget)):
        eligible = np.asarray(mask, dtype=bool) & (action < caps)
        if not np.any(eligible):
            break
        marginal = np.full(base.n_nodes, -np.inf, dtype=np.float64)
        marginal[eligible] = score[eligible] / (action[eligible] + 1.0)
        action[int(np.argmax(marginal))] += 1
    return action


def primal_dual_cluster_action(base, mask, caps, statistics: ClusterStatistics, budget: int):
    statistics.update_duals()
    action = np.zeros(base.n_nodes, dtype=np.int64)
    members = np.flatnonzero(mask)
    if not len(members):
        return action
    queue = base.queue.astype(np.float64) / max(1, base.cfg.queue_max_packets)
    expiring = np.asarray([
        sum(age >= base.cfg.packet_ttl_rounds for age in ages)
        for ages in base.packet_ages
    ], dtype=np.float64)
    expiring /= max(1.0, float(expiring.max(initial=0.0)))
    ratios = np.divide(statistics.delivered, np.maximum(statistics.offered, 1.0))
    service_deficit = np.maximum(0.0, ratios[members].max(initial=0.0) - ratios)
    energy = np.clip(base.energy / base.cfg.initial_energy_j, 0.0, 1.0)
    forecast = np.maximum(base._state()[:, 1], 0.0)
    forecast /= max(1e-12, float(forecast.max(initial=0.0)))
    config = statistics.dual_config
    score = (
        statistics.dual["delivery"] * queue
        + statistics.dual["stale"] * expiring
        + statistics.dual["fairness"] * service_deficit
        + float(config["energy_weight"]) * energy
        + float(config["harvest_weight"]) * forecast
    )
    for _ in range(int(budget)):
        eligible = np.asarray(mask, dtype=bool) & (action < caps)
        if not np.any(eligible):
            break
        marginal = np.full(base.n_nodes, -np.inf, dtype=np.float64)
        marginal[eligible] = score[eligible] / (action[eligible] + 1.0)
        action[int(np.argmax(marginal))] += 1
    return action


def learned_cluster_action(agent, base, risk, ch, members, mask, statistics, policy, budget):
    physical = base._state()
    observation = build_step3_observation(
        base, physical, mask, ch=ch, members=members, risk_config=risk,
        schema=STEP3_CH_CONTEXT_SCHEMA, wrapper=statistics,
    )
    caps = np.minimum(base.queue, base.cfg.n_max).astype(np.int64)
    caps[~mask] = 0
    if policy == "hta_no_ch_context":
        # Context is inserted immediately before the 32-dimensional embedding.
        start = int(agent.cfg.embedding_start_dim) - len(CH_CONTEXT_NAMES)
        observation[:, start:int(agent.cfg.embedding_start_dim)] = 0.0
    if policy != "hta_no_set_context":
        action, q_values = agent.act(
            observation, mask, epsilon=0.0, caps=caps, budget=budget,
            tie_break_priorities=np.arange(base.n_nodes),
        )
        return action, q_values, caps

    if not hasattr(agent.online, "q_values_without_set_context"):
        raise RuntimeError("set-context intervention requires equivariant set architecture")
    agent.online.eval()
    with torch.no_grad():
        state_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.as_tensor(mask, dtype=torch.bool).unsqueeze(0)
        transformed = agent._transform_state_tensor(state_tensor)
        q_values = agent.online.q_values_without_set_context(
            transformed, mask_tensor
        )[0].cpu().numpy()
    action = agent._project(
        q_values, mask, caps=caps, budget=budget,
        tie_break_priorities=np.arange(base.n_nodes),
    )
    return action, q_values, caps


def residual_energy_fairness(energy: np.ndarray, alive: np.ndarray) -> float:
    values = np.asarray(energy, dtype=np.float64)[np.asarray(alive, dtype=bool)]
    if not len(values):
        return 0.0
    denominator = len(values) * np.square(values).sum()
    return float(values.sum() ** 2 / denominator) if denominator > 0.0 else 0.0


def jain_weighted_goodput_efficiency(packets_per_j: float, service_fairness: float) -> float:
    """Reliable network goodput/J discounted by unequal node service.

    This remains in packets/J because Jain's index is dimensionless.  It is a
    secondary diagnostic; raw goodput/J and Jain fairness must also be reported.
    """
    efficiency = float(packets_per_j)
    fairness = float(service_fairness)
    if not math.isfinite(efficiency) or efficiency < 0.0:
        raise ValueError("packets_per_j must be finite and non-negative")
    if not math.isfinite(fairness) or not 0.0 <= fairness <= 1.0:
        raise ValueError("service_fairness must lie in [0, 1]")
    return efficiency * fairness


def evaluate_trial(
    contract_path: str, scenario: dict, policy: str, seed: int, smoke: bool,
    checkpoint_override: str | None = None,
):
    contract = load_contract(resolve(contract_path))
    torch.set_num_threads(int(contract["threads_per_worker"]))
    horizon = min(int(contract["smoke_horizon"]), int(contract["horizon"])) if smoke else int(contract["horizon"])
    base, _, risk = build_environment(contract, scenario, int(seed), horizon)
    qos = Step3QoSConstraintConfig.from_payload(_load_json(contract["qos_config"]))
    checkpoint_path = checkpoint_override or contract["source_checkpoint"]
    agent = load_agent(resolve(checkpoint_path))[0] if policy.startswith("hta_") else None
    cluster_count = len(base.cluster_heads)
    statistics = [ClusterStatistics(base.n_nodes, qos, contract["primal_dual"]) for _ in range(cluster_count)]
    offered = np.zeros(base.n_nodes, dtype=np.int64)
    delivered = np.zeros(base.n_nodes, dtype=np.int64)
    stale = 0
    consumed = 0.0
    allocated = 0
    decision_ns = []
    projection_rows = []
    fnd = hnd = lnd = None
    feasibility_violations = 0

    while base.round < horizon and np.any(base.alive):
        combined = np.zeros(base.n_nodes, dtype=np.int64)
        step_records = []
        for cluster in range(cluster_count):
            ch, members, mask = cluster_members(base, cluster)
            caps = np.minimum(base.queue, base.cfg.n_max).astype(np.int64)
            caps[~mask] = 0
            started = time.perf_counter_ns()
            q_values = None
            if policy.startswith("hta_"):
                cluster_action, q_values, caps = learned_cluster_action(
                    agent, base, risk, ch, members, mask, statistics[cluster],
                    policy, int(contract["budget"]),
                )
            elif policy == "energy_proportional":
                cluster_action = energy_proportional_cluster_action(
                    base, mask, caps, int(contract["budget"]),
                    float(contract["energy_proportional_score_exponent"]),
                )
            elif policy == "online_primal_dual":
                cluster_action = primal_dual_cluster_action(
                    base, mask, caps, statistics[cluster], int(contract["budget"])
                )
            else:
                raise ValueError(policy)
            decision_ns.append(time.perf_counter_ns() - started)
            if int(cluster_action.sum()) > int(contract["budget"]) or np.any(cluster_action > caps):
                feasibility_violations += 1
            combined += cluster_action
            if q_values is not None:
                projection_rows.append(projection_optimality_diagnostic(
                    q_values, cluster_action, int(contract["budget"]), caps=caps
                ))
            queue_before = base.queue.copy()
            expiring_before = np.asarray([
                sum(age >= base.cfg.packet_ttl_rounds for age in ages)
                for ages in base.packet_ages
            ], dtype=np.int64)
            step_records.append((cluster, mask, queue_before, expiring_before, cluster_action))

        state, _, terminated, truncated, info = base.step(combined)
        del state
        consumed += float(np.asarray(info["energy_trace"]["consumed"]).sum())
        allocated += int(combined.sum())
        for cluster, mask, queue_before, expiring_before, cluster_action in step_records:
            served = np.minimum(queue_before, cluster_action)
            offered[mask] += queue_before[mask]
            delivered[mask] += served[mask]
            stale_now = np.maximum(0, expiring_before[mask] - served[mask])
            stale += int(stale_now.sum())
            stats = statistics[cluster]
            stats.offered[mask] += queue_before[mask]
            stats.delivered[mask] += served[mask]
            stats.stale += int(stale_now.sum())
            per_member = float(queue_before[mask].sum()) / max(1, int(mask.sum()))
            stats.step3_demand_ewma = 0.8 * stats.step3_demand_ewma + 0.2 * per_member
        alive_count = int(base.alive.sum())
        if fnd is None and alive_count < base.n_nodes:
            fnd = int(base.round)
        if hnd is None and alive_count <= base.n_nodes // 2:
            hnd = int(base.round)
        if lnd is None and alive_count == 0:
            lnd = int(base.round)
        if terminated or truncated:
            break

    demand = max(1, int(offered.sum()))
    projection = {
        "comparisons": len(projection_rows),
        "allocation_match_rate": float(np.mean([row["allocation_match"] for row in projection_rows])) if projection_rows else None,
        "mean_absolute_regret": float(np.mean([row["absolute_regret"] for row in projection_rows])) if projection_rows else None,
        "max_absolute_regret": float(max((row["absolute_regret"] for row in projection_rows), default=0.0)),
    }
    service_fairness = episode_service_fairness(delivered, offered)
    packets_per_j = int(base.total_packets) / max(consumed, 1e-12)
    qos_joint_pass = bool(
        int(delivered.sum()) / demand >= qos.minimum_delivery_ratio
        and stale / demand <= qos.maximum_stale_drop_ratio
        and service_fairness >= qos.minimum_queue_fairness
    )
    result = {
        "scenario": scenario["id"], "policy": policy, "seed": int(seed),
        "network_wide_simultaneous_control": True,
        "steps": int(base.round), "nodes": int(base.n_nodes),
        "delivery_ratio": int(delivered.sum()) / demand,
        "stale_ratio": stale / demand,
        "service_fairness": service_fairness,
        "qos_joint_pass": qos_joint_pass,
        "fnd_round": int(fnd if fnd is not None else horizon),
        "hnd_round": int(hnd if hnd is not None else horizon),
        "lnd_round": int(lnd if lnd is not None else horizon),
        "fnd_observed": fnd is not None, "hnd_observed": hnd is not None,
        "lnd_observed": lnd is not None,
        "global_packets": int(base.total_packets), "network_energy_j": consumed,
        "packets_per_j": packets_per_j,
        "jain_weighted_goodput_per_j": jain_weighted_goodput_efficiency(
            packets_per_j, service_fairness
        ),
        "allocated_slots": allocated,
        "residual_energy_fairness": residual_energy_fairness(base.energy, base.alive),
        "alive_nodes_final": int(base.alive.sum()),
        "decision_latency_ms_per_cluster_mean": float(np.mean(decision_ns) / 1e6),
        "decision_latency_ms_per_network_round_mean": float(np.sum(decision_ns) / max(1, base.round) / 1e6),
        "feasibility_violations": int(feasibility_violations),
        "projection_optimality": projection,
    }
    return scenario["id"], policy, int(seed), result


def paired_summary(values: np.ndarray, rng, resamples: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    draws = rng.choice(values, size=(int(resamples), len(values)), replace=True).mean(axis=1)
    p_value = 1.0 if np.allclose(values, 0.0) else float(
        wilcoxon(values, alternative="two-sided", method="auto").pvalue
    )
    return {
        "pairs": len(values), "mean_difference": float(values.mean()),
        "median_difference": float(np.median(values)),
        "bootstrap_95_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "wilcoxon_two_sided_p": p_value,
    }


def retention_summary(values: np.ndarray, rng, resamples: int) -> dict:
    values = np.asarray(values, dtype=np.float64)
    draws = rng.choice(values, size=(int(resamples), len(values)), replace=True).mean(axis=1)
    centered = values - 1.0
    p_value = 1.0 if np.allclose(centered, 0.0) else float(
        wilcoxon(centered, alternative="two-sided", method="auto").pvalue
    )
    return {
        "pairs": len(values), "mean_retention": float(values.mean()),
        "median_retention": float(np.median(values)),
        "bootstrap_95_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "wilcoxon_two_sided_vs_unit_retention_p": p_value,
    }


def analyze_load_robust_fair_energy(contract: dict, raw: dict, seeds: list[int], rng) -> dict:
    reference_id = contract["derived_metrics"]["load_robust_fair_energy_retention"]["reference_scenario"]
    stress_id = contract["derived_metrics"]["load_robust_fair_energy_retention"]["stress_scenario"]
    if reference_id not in raw or stress_id not in raw:
        return {"assessed": False, "reason": "required paired scenarios were not evaluated"}
    per_policy = {}
    ratios = {}
    for policy in contract["policies"]:
        ratios[policy] = np.asarray([
            raw[stress_id][policy][str(seed)]["jain_weighted_goodput_per_j"]
            / max(raw[reference_id][policy][str(seed)]["jain_weighted_goodput_per_j"], 1e-12)
            for seed in seeds
        ], dtype=np.float64)
        per_policy[policy] = {
            **retention_summary(ratios[policy], rng, int(contract["bootstrap_resamples"])),
            "per_seed": {str(seed): float(value) for seed, value in zip(seeds, ratios[policy])},
        }
    ranking = sorted(
        contract["policies"], key=lambda policy: per_policy[policy]["mean_retention"], reverse=True
    )
    full_minus = {
        policy: paired_summary(
            ratios["hta_mac"] - ratios[policy], rng, int(contract["bootstrap_resamples"])
        )
        for policy in contract["policies"] if policy != "hta_mac"
    }
    return {
        "assessed": True,
        "equation": "JWGE(traffic_high) / JWGE(reference_100), paired by environment seed",
        "pilot_derived_requires_sealed_confirmation": True,
        "policies": per_policy,
        "rank_by_mean_retention": ranking,
        "hta_mac_minus_alternative": full_minus,
    }


def analyze(contract: dict, raw: dict, seeds: list[int], scenarios: list[dict]) -> dict:
    rng = np.random.default_rng(int(contract["bootstrap_seed"]))
    metrics = (
        "delivery_ratio", "stale_ratio", "service_fairness", "fnd_round",
        "hnd_round", "lnd_round", "packets_per_j", "network_energy_j",
        "jain_weighted_goodput_per_j", "residual_energy_fairness",
        "decision_latency_ms_per_network_round_mean",
    )
    output = {}
    for scenario in scenarios:
        identifier = scenario["id"]
        output[identifier] = {"policies": {}, "full_minus_alternative": {}}
        for policy in contract["policies"]:
            rows = [raw[identifier][policy][str(seed)] for seed in seeds]
            output[identifier]["policies"][policy] = {
                **{f"mean_{metric}": float(np.mean([row[metric] for row in rows])) for metric in metrics},
                "fnd_events": int(sum(row["fnd_observed"] for row in rows)),
                "hnd_events": int(sum(row["hnd_observed"] for row in rows)),
                "lnd_events": int(sum(row["lnd_observed"] for row in rows)),
                "qos_joint_passes": int(sum(row["qos_joint_pass"] for row in rows)),
                "feasibility_violations": int(sum(row["feasibility_violations"] for row in rows)),
            }
        full = raw[identifier]["hta_mac"]
        for policy in contract["policies"]:
            if policy == "hta_mac":
                continue
            comparisons = {}
            for metric in metrics:
                differences = np.asarray([
                    full[str(seed)][metric] - raw[identifier][policy][str(seed)][metric]
                    for seed in seeds
                ])
                comparisons[metric] = paired_summary(
                    differences, rng, int(contract["bootstrap_resamples"])
                )
            output[identifier]["full_minus_alternative"][policy] = comparisons
    output["cross_scenario_load_robustness"] = analyze_load_robust_fair_energy(
        contract, raw, seeds, rng
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.pilot:
        raise ValueError("--smoke and --pilot are mutually exclusive")
    contract_path = resolve(args.contract)
    output_path = resolve(args.output)
    contract = load_contract(contract_path)
    seeds = (
        [int(contract["smoke_seed"])] if args.smoke
        else list(map(int, contract["pilot_seeds"])) if args.pilot
        else list(map(int, contract["evaluation_seeds"]))
    )
    scenarios = (
        contract["scenarios"][:2] if args.smoke
        else [row for row in contract["scenarios"] if row["id"] in {"reference_100", "nodes_20", "traffic_high"}]
        if args.pilot else contract["scenarios"]
    )
    raw = {row["id"]: {policy: {} for policy in contract["policies"]} for row in scenarios}
    started = time.perf_counter()
    jobs = [
        (scenario, policy, seed)
        for scenario in scenarios
        for policy in contract["policies"]
        for seed in seeds
    ]
    if args.smoke:
        for index, (scenario_spec, policy, seed) in enumerate(jobs, start=1):
            scenario, completed_policy, completed_seed, row = evaluate_trial(
                str(contract_path), scenario_spec, policy, seed, True
            )
            raw[scenario][completed_policy][str(completed_seed)] = row
            print(f"PUBLICATION_EXTENSION_PROGRESS={index}/{len(jobs)}", flush=True)
    else:
        tasks = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
            for scenario, policy, seed in jobs:
                tasks.append(pool.submit(
                    evaluate_trial, str(contract_path), scenario, policy, seed, False
                ))
            for index, future in enumerate(concurrent.futures.as_completed(tasks), start=1):
                scenario, policy, seed, row = future.result()
                raw[scenario][policy][str(seed)] = row
                print(f"PUBLICATION_EXTENSION_PROGRESS={index}/{len(tasks)}", flush=True)
    checks = {
        "all_tasks_complete": all(
            set(raw[row["id"]][policy]) == set(map(str, seeds))
            for row in scenarios for policy in contract["policies"]
        ),
        "zero_feasibility_violations": all(
            item["feasibility_violations"] == 0
            for scenario in raw.values() for policy in scenario.values() for item in policy.values()
        ),
        "network_wide_intervention": all(
            item["network_wide_simultaneous_control"]
            for scenario in raw.values() for policy in scenario.values() for item in policy.values()
        ),
    }
    payload = {
        "schema_version": 1,
        "status": "publication_extension_smoke_complete" if args.smoke else "publication_extension_pilot_complete" if args.pilot else "publication_extension_evaluation_complete",
        "smoke": bool(args.smoke), "pilot": bool(args.pilot), "contract": str(contract_path),
        "contract_sha256": sha256(contract_path), "runner_sha256": sha256(Path(__file__)),
        "seeds": seeds, "scenarios": [row["id"] for row in scenarios],
        "checks": checks, "raw": raw,
        "analysis": {} if args.smoke else analyze(contract, raw, seeds, scenarios),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": contract["claim_boundary"],
    }
    if not all(checks.values()):
        payload["status"] = "publication_extension_incomplete"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "checks": checks, "output": str(output_path)}, indent=2))
    return 0 if all(checks.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
