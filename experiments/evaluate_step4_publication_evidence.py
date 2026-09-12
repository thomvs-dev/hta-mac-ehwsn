"""Bounded development evaluation for the four publication-evidence gaps.

This script deliberately separates the primary frozen HEART-CH environment
from an independently generated balanced-rotation transfer environment.
Development results are descriptive and cannot be used for model selection on
the reserved 3900--3919 confirmation cohort.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.ch_depletion_risk import validate_ch_risk_config
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from envs.step3_lifetime_env import RoleSeparatedScheduledMACEnv, configure_step3_risk
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from envs.step3_v3_env import Step3V3DynamicClusterTrainingEnv
from experiments.diagnose_step3_delivery_feasibility import build_environments, load_agent
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.evaluate_step3_final_matched_baselines import (
    energy_proportional_action,
    load_ranker,
)
from experiments.paper_aligned_environment import _spatial_embedding
from experiments.run_phase3_pilot import build_assets
from experiments.sweep_step3_primary_listwise_residual import (
    apply_ranker,
    base_and_added,
    removal_count,
)


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path: str | Path) -> str:
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def load_contract(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen_before_step4_publication_development":
        raise RuntimeError("Step 4 development contract is not frozen")
    for field in ("source_checkpoint", "selected_ranker_checkpoint", "qos_config", "risk_config"):
        if sha256(payload[field]) != payload[f"{field}_sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {field}")
    development = set(payload["development_seeds"])
    if development.intersection(payload["prohibited_prior_seeds"]):
        raise RuntimeError("Step 4 development seeds overlap prior evidence")
    if development.intersection(payload["confirmation_seeds"]):
        raise RuntimeError("development and confirmation cohorts overlap")
    if payload.get("confirmation_seeds_opened") is not False:
        raise RuntimeError("confirmation cohort must remain sealed")
    if int(payload["workers"]) * int(payload["threads_per_worker"]) > 16:
        raise RuntimeError("CPU contract exceeds 16 logical workers")
    identifiers = [row["id"] for row in payload["pareto_operating_points"]]
    if len(identifiers) != len(set(identifiers)) or len(identifiers) != 3:
        raise RuntimeError("exactly three unique Pareto points are required")
    scenarios = [row["id"] for row in payload["transfer_scenarios"]]
    if len(scenarios) != len(set(scenarios)):
        raise RuntimeError("transfer scenario identifiers must be unique")
    return payload


class ScenarioRoleSeparatedScheduledMACEnv(RoleSeparatedScheduledMACEnv):
    """Evaluation-only environment with explicit traffic/harvest interventions."""

    def __init__(
        self,
        *args,
        arrival_rate: float = 1.0,
        harvest_multiplier: float = 1.0,
        trace_w_m2: np.ndarray | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if arrival_rate <= 0.0 or harvest_multiplier <= 0.0:
            raise ValueError("arrival rate and harvest multiplier must be positive")
        self.arrival_rate = float(arrival_rate)
        self.harvest_multiplier = float(harvest_multiplier)
        self.trace_w_m2 = None if trace_w_m2 is None else np.asarray(trace_w_m2, dtype=np.float64)
        if self.trace_w_m2 is not None:
            if self.trace_w_m2.ndim != 1 or len(self.trace_w_m2) < 24:
                raise ValueError("external trace must contain at least 24 scalar observations")
            if np.any(~np.isfinite(self.trace_w_m2)) or np.any(self.trace_w_m2 < 0.0):
                raise ValueError("external irradiance trace must be finite and nonnegative")
            positive = self.trace_w_m2[self.trace_w_m2 > 0.0]
            if not len(positive):
                raise ValueError("external irradiance trace contains no positive observations")
            self.trace_positive_mean = float(positive.mean())
        else:
            self.trace_positive_mean = None

    def reset(self, *, seed: int, frozen_snapshot: dict):
        result = super().reset(seed=seed, frozen_snapshot=frozen_snapshot)
        trace_rng = np.random.default_rng(int(seed) + 700_001)
        self.trace_node_factor = trace_rng.uniform(0.90, 1.10, size=self.n_nodes)
        return result

    def _sample_harvest(self) -> np.ndarray:
        hmm_sample = super()._sample_harvest()
        if self.trace_w_m2 is None:
            return hmm_sample * self.harvest_multiplier
        alive = np.flatnonzero(self.alive)
        result = np.zeros(self.n_nodes, dtype=np.float64)
        if len(alive):
            trace_value = float(self.trace_w_m2[self.round % len(self.trace_w_m2)])
            positive_hmm = hmm_sample[alive][hmm_sample[alive] > 0.0]
            # Frozen HMM mean provides the joule scale; the trace supplies temporal shape.
            reference_j = float(positive_hmm.mean()) if len(positive_hmm) else max(
                1e-12, float(np.mean(self.solar.mean) * self.cfg.solar_scale)
            )
            result[alive] = (
                trace_value / self.trace_positive_mean
                * reference_j
                * self.trace_node_factor[alive]
                * self.harvest_multiplier
            )
        return result

    def _update_queues(self, delivered: np.ndarray) -> None:
        for node in range(self.n_nodes):
            served = min(int(delivered[node]), len(self.packet_ages[node]))
            if served:
                del self.packet_ages[node][:served]
            if not self.alive[node]:
                self.dropped_death_packets += len(self.packet_ages[node])
                self.packet_ages[node] = []
                continue
            aged = [age + 1 for age in self.packet_ages[node]]
            retained = [age for age in aged if age <= self.cfg.packet_ttl_rounds]
            self.dropped_stale_packets += len(aged) - len(retained)
            arrivals = int(self.np_random.poisson(self.arrival_rate))
            retained.extend([0] * arrivals)
            self.total_packets_generated += arrivals
            if len(retained) > self.cfg.queue_max_packets:
                overflow = len(retained) - self.cfg.queue_max_packets
                self.dropped_overflow_packets += overflow
                retained = retained[overflow:]
            self.packet_ages[node] = retained
        self.queue = np.asarray([len(packets) for packets in self.packet_ages], dtype=np.int64)
        self.max_backlog_observed = max(self.max_backlog_observed, int(self.queue.max(initial=0)))


def independent_schedule_bundle(
    *, seed: int, nodes: int, horizon: int, field_size: float,
    bs_position: tuple[float, float], solar_states: int, thermal_states: int,
) -> dict:
    if nodes < 4 or nodes > 100:
        raise ValueError("independent transfer supports 4--100 nodes")
    rng = np.random.default_rng(int(seed) + 91_003)
    positions = rng.uniform(0.0, float(field_size), size=(nodes, 2))
    embedding = _spatial_embedding(positions, field_size, bs_position)
    solar = rng.integers(0, solar_states, size=nodes, dtype=np.int64)
    thermal = rng.integers(0, thermal_states, size=nodes, dtype=np.int64)
    ch_count = max(1, int(round(nodes * 0.05)))
    epoch = int(math.ceil(nodes / ch_count))
    permutation = rng.permutation(nodes)
    frames = []
    for round_index in range(horizon):
        if round_index % epoch == 0:
            permutation = rng.permutation(nodes)
        start = (round_index % epoch) * ch_count
        indices = np.arange(start, start + ch_count) % nodes
        heads = np.sort(permutation[indices]).astype(np.int64)
        frames.append({
            "positions": positions.copy(),
            "cluster_heads": heads,
            "solar_states": solar.copy(),
            "thermal_states": thermal.copy(),
            "stgcn_embedding": embedding.copy(),
        })
    return {
        **frames[0],
        "schedule": frames,
        "schedule_metadata": {
            "schedule_schema_version": "step4_independent_balanced_rotation_v1",
            "coverage_rounds": horizon,
            "complete": True,
            "generator": "independent_balanced_rotation",
            "num_nodes": nodes,
            "cluster_heads_per_round": ch_count,
        },
    }


def load_trace(path: Path) -> np.ndarray:
    values = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            values.append(float(row["global_in_plane_irradiance_w_m2"]))
    result = np.asarray(values, dtype=np.float64)
    if len(result) < 24 * 300:
        raise RuntimeError("external trace does not cover at least 300 days")
    return result


def build_transfer_environments(contract: dict, scenario: dict, trace: np.ndarray | None = None):
    horizon = int(contract["horizon"])
    _, solar, thermal, radio, base_cfg, _ = build_assets(horizon)
    risk = validate_ch_risk_config(json.loads(resolve(contract["risk_config"]).read_text()))
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
    environments = []
    observation_schema = contract.get(
        "observation_schema", STEP3_CH_CONTEXT_SCHEMA
    )
    for seed in contract["development_seeds"]:
        bundle = independent_schedule_bundle(
            seed=int(seed), nodes=nodes, horizon=horizon, field_size=field_size,
            bs_position=bs_position, solar_states=len(solar.initial), thermal_states=len(thermal.initial),
        )
        ranks = len(bundle["schedule"][0]["cluster_heads"])
        for rank in range(ranks):
            base = ScenarioRoleSeparatedScheduledMACEnv(
                cfg, radio, solar, thermal, idle_energy_enabled=True,
                arrival_rate=float(scenario["arrival_rate"]),
                harvest_multiplier=float(scenario["harvest_multiplier"]),
                trace_w_m2=trace,
            )
            env = Step3V3DynamicClusterTrainingEnv(
                base, bundle, seed=int(seed), target_rank=rank,
                observation_schema=observation_schema, risk_config=risk,
            )
            env.reset()
            environments.append(env)
    return environments


class OnlinePrimalDualQoS:
    """Online dual-ascent allocator; a constrained-learning, non-neural baseline."""

    def __init__(self, qos: Step3QoSConstraintConfig, config: dict, budget: int):
        self.qos = qos
        self.config = config
        self.budget = int(budget)
        self.reset()

    def reset(self):
        initial = self.config["initial_multiplier"]
        self.dual = {name: float(initial[name]) for name in ("delivery", "stale", "fairness")}

    def _update_duals(self, env) -> None:
        counts = env.step3_qos_counts
        demand = max(1, int(counts["demand"]))
        delivery = int(counts["delivered"]) / demand
        stale = int(counts["stale"]) / demand
        fairness = float(counts["episode_service_fairness"])
        residuals = {
            "delivery": self.qos.minimum_delivery_ratio - delivery,
            "stale": stale - self.qos.maximum_stale_drop_ratio,
            "fairness": self.qos.minimum_queue_fairness - fairness,
        }
        eta, maximum = float(self.config["learning_rate"]), float(self.config["maximum_multiplier"])
        for name, residual in residuals.items():
            self.dual[name] = float(np.clip(self.dual[name] + eta * residual, 0.0, maximum))

    def action(self, env, mask, caps) -> np.ndarray:
        self._update_duals(env)
        mask = np.asarray(mask, dtype=bool)
        caps = np.asarray(caps, dtype=np.int64)
        action = np.zeros_like(caps)
        members = np.flatnonzero(mask)
        if not len(members):
            return action
        queue = env.base.queue.astype(np.float64) / max(1, env.base.cfg.queue_max_packets)
        expiring = np.asarray([
            sum(age >= env.base.cfg.packet_ttl_rounds for age in ages)
            for ages in env.base.packet_ages
        ], dtype=np.float64)
        expiring /= max(1.0, float(expiring.max(initial=0.0)))
        offered = env.step3_episode_offered_per_node.astype(np.float64)
        delivered = env.step3_episode_delivered_per_node.astype(np.float64)
        ratios = np.divide(delivered, np.maximum(offered, 1.0))
        service_deficit = np.maximum(0.0, ratios[members].max(initial=0.0) - ratios)
        energy = np.clip(env.base.energy / env.base.cfg.initial_energy_j, 0.0, 1.0)
        forecast = np.maximum(env.base._state()[:, 1], 0.0)
        forecast /= max(1e-12, float(forecast.max(initial=0.0)))
        score = (
            self.dual["delivery"] * queue
            + self.dual["stale"] * expiring
            + self.dual["fairness"] * service_deficit
            + float(self.config["energy_weight"]) * energy
            + float(self.config["harvest_weight"]) * forecast
        )
        for _ in range(self.budget):
            eligible = mask & (action < caps)
            if not np.any(eligible):
                break
            marginal = np.full(len(action), -np.inf, dtype=np.float64)
            marginal[eligible] = score[eligible] / (action[eligible] + 1.0)
            node = int(np.argmax(marginal))
            if not np.isfinite(marginal[node]):
                break
            action[node] += 1
        return action


def evaluate_rows(contract: dict, environments, policy: str, band: dict | None = None) -> list[dict]:
    qos = Step3QoSConstraintConfig.from_payload(json.loads(resolve(contract["qos_config"]).read_text()))
    agent, _ = load_agent(resolve(contract["source_checkpoint"]))
    ranker = load_ranker(resolve(contract["selected_ranker_checkpoint"]))
    runtime = dict(contract)
    runtime["band"] = dict(band or next(
        row for row in contract["pareto_operating_points"] if row["id"] == "balanced_frozen"
    ))
    rows = []
    for env in environments:
        observation, mask, _ = env.reset()
        dual = OnlinePrimalDualQoS(qos, contract["primal_dual"], int(contract["budget"]))
        done, energy, allocated, steps = False, 0.0, 0, 0
        while not done:
            padded, padded_mask, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
            if policy == "learned_listwise_residual":
                _, padded_mask, caps, _, q_values, added = base_and_added(
                    agent, env, observation, mask, runtime
                )
                count = removal_count(added, env, runtime["band"])
                action = apply_ranker(ranker, env, q_values, added, caps, padded_mask, count)
            elif policy == "raw_branching_c51":
                action, _ = agent.act(
                    padded, padded_mask, epsilon=0.0, caps=caps,
                    budget=int(contract["budget"]), tie_break_priorities=np.arange(env.base.n_nodes),
                )
            elif policy == "energy_proportional_tuned":
                action = energy_proportional_action(
                    env, padded_mask, int(contract["budget"]),
                    float(contract["energy_proportional_score_exponent"]),
                )
            elif policy == "online_primal_dual_qos":
                action = dual.action(env, padded_mask, caps)
            else:
                raise ValueError(policy)
            allocated += int(action.sum())
            observation, mask, done, info = env.step(action)
            energy += float(np.asarray(info["energy_trace"]["consumed"]).sum())
            steps += 1
        counts = env.step3_qos_counts
        demand = max(1, int(counts["demand"]))
        delivery = int(counts["delivered"]) / demand
        stale = int(counts["stale"]) / demand
        fairness = float(counts["episode_service_fairness"])
        event = env.base.t_fnd is not None
        rows.append({
            "policy": policy,
            "seed": int(env.seed),
            "target_rank": int(env.target_rank),
            "delivery_ratio": delivery,
            "stale_ratio": stale,
            "episode_service_fairness": fairness,
            "joint_qos_pass": bool(
                delivery >= qos.minimum_delivery_ratio
                and stale <= qos.maximum_stale_drop_ratio
                and fairness >= qos.minimum_queue_fairness
            ),
            "fnd_event_observed": bool(event),
            "fnd_or_censoring_round": int(env.base.t_fnd if event else steps),
            "restricted_survival_rounds": int(min(env.base.t_fnd if event else steps, contract["horizon"])),
            "global_packets_delivered": int(env.base.total_packets),
            "network_energy_j": energy,
            "global_packets_per_j": int(env.base.total_packets) / max(energy, 1e-12),
            "allocated_slots": allocated,
            "steps": steps,
        })
    return rows


def aggregate(rows: list[dict]) -> dict:
    metrics = (
        "delivery_ratio", "stale_ratio", "episode_service_fairness",
        "restricted_survival_rounds", "global_packets_delivered",
        "network_energy_j", "global_packets_per_j", "allocated_slots",
    )
    return {
        "rows": len(rows),
        "seed_units": len({row["seed"] for row in rows}),
        "joint_qos_pass_count": int(sum(row["joint_qos_pass"] for row in rows)),
        "joint_qos_pass_rate": float(np.mean([row["joint_qos_pass"] for row in rows])),
        "fnd_event_count": int(sum(row["fnd_event_observed"] for row in rows)),
        "fnd_censored_count": int(sum(not row["fnd_event_observed"] for row in rows)),
        **{f"mean_{name}": float(np.mean([row[name] for row in rows])) for name in metrics},
    }


def pareto_worker(contract_path: str, point: dict) -> tuple[str, dict]:
    contract = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), 20260815 + len(point["id"]))
    risk = validate_ch_risk_config(json.loads(resolve(contract["risk_config"]).read_text()))
    envs, _, _ = build_environments(None, risk, int(contract["horizon"]), seeds=contract["development_seeds"])
    rows = evaluate_rows(contract, envs, "learned_listwise_residual", point)
    return point["id"], {"operating_point": point, "aggregate": aggregate(rows), "rows": rows}


def transfer_worker(contract_path: str, scenario: dict, policy: str, trace_path: str | None):
    contract = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), 20260815 + len(scenario["id"]) + len(policy))
    trace = load_trace(Path(trace_path)) if trace_path else None
    envs = build_transfer_environments(contract, scenario, trace=trace)
    rows = evaluate_rows(contract, envs, policy)
    return scenario["id"], policy, {"aggregate": aggregate(rows), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="use one seed and at most 40 rounds")
    args = parser.parse_args()
    contract_path, output_path = resolve(args.contract), resolve(args.output)
    contract = load_contract(contract_path)
    if args.smoke:
        contract = dict(contract)
        contract["development_seeds"] = contract["development_seeds"][:1]
        contract["horizon"] = min(40, int(contract["horizon"]))
        smoke_path = output_path.parent / "step4_smoke_runtime_contract.json"
        smoke_path.parent.mkdir(parents=True, exist_ok=True)
        smoke_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        contract_path = smoke_path

    started = time.perf_counter()
    pareto = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(3, int(contract["workers"]))) as pool:
        futures = [pool.submit(pareto_worker, str(contract_path), point) for point in contract["pareto_operating_points"]]
        for future in concurrent.futures.as_completed(futures):
            identifier, result = future.result()
            pareto[identifier] = result
            print(f"PARETO_COMPLETE={identifier} ROWS={result['aggregate']['rows']}", flush=True)

    scenarios = [dict(row) for row in contract["transfer_scenarios"]]
    trace_info = dict(contract["external_trace"])
    trace_path = resolve(trace_info["path"])
    trace_metadata_path = trace_path.with_suffix(".metadata.json")
    if trace_path.is_file() and trace_metadata_path.is_file():
        metadata = json.loads(trace_metadata_path.read_text(encoding="utf-8"))
        actual_hash = sha256(trace_path)
        if actual_hash != metadata.get("trace_sha256"):
            raise RuntimeError("external trace hash differs from its metadata")
        if trace_info.get("sha256") and actual_hash != trace_info["sha256"]:
            raise RuntimeError("external trace hash differs from frozen Step 4 contract")
        trace_info.update({"status": "available_and_verified", "sha256": actual_hash, "metadata": metadata})
        scenarios.append({
            "id": "external_trace_srrl_2020", "nodes": 100, "field_scale": 1.0,
            "arrival_rate": 1.0, "harvest_multiplier": 1.0, "initial_energy_multiplier": 1.0,
            "trace_path": str(trace_path),
        })
    else:
        trace_info["status"] = "blocked_missing_external_trace"

    transfer = {scenario["id"]: {} for scenario in scenarios}
    tasks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
        for scenario in scenarios:
            for policy in contract["transfer_policies"]:
                tasks.append(pool.submit(
                    transfer_worker, str(contract_path), scenario, policy, scenario.get("trace_path")
                ))
        for future in concurrent.futures.as_completed(tasks):
            scenario_id, policy, result = future.result()
            transfer[scenario_id][policy] = result
            print(f"TRANSFER_COMPLETE={scenario_id} POLICY={policy} ROWS={result['aggregate']['rows']}", flush=True)

    checks = {
        "development_cohort_only": not set(contract["development_seeds"]).intersection(contract["confirmation_seeds"]),
        "pareto_complete": set(pareto) == {row["id"] for row in contract["pareto_operating_points"]},
        "transfer_complete": all(set(policies) == set(contract["transfer_policies"]) for policies in transfer.values()),
        "external_trace_complete": trace_info["status"] == "available_and_verified",
        "no_confirmation_seed_opened": True,
    }
    payload = {
        "schema_version": 1,
        "status": "step4_development_complete" if all(checks.values()) else "step4_development_partial",
        "smoke": bool(args.smoke),
        "contract_sha256": sha256(contract_path),
        "evaluator_sha256": sha256(Path(__file__)),
        "development_seeds": contract["development_seeds"],
        "confirmation_seeds_opened": False,
        "pareto": pareto,
        "transfer": transfer,
        "external_trace": trace_info,
        "checks": checks,
        "publication_evidence": False,
        "selection_or_retuning_performed": False,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": contract["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "checks": checks,
        "elapsed_seconds": payload["elapsed_seconds"], "output": str(output_path),
    }, indent=2), flush=True)
    return 0 if all(checks.values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
