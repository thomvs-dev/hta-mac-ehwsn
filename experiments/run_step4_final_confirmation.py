"""Run the frozen one-shot 20-seed HTA-MAC final confirmation."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from experiments.diagnose_step3_delivery_feasibility import load_agent
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.evaluate_step3_final_matched_baselines import energy_proportional_action
from experiments.evaluate_step4_publication_evidence import (
    OnlinePrimalDualQoS,
    build_transfer_environments,
    load_trace,
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(value: str | Path) -> str:
    return hashlib.sha256(resolve(value).read_bytes()).hexdigest()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_final_20seed_confirmation":
        raise RuntimeError("final confirmation contract is not frozen")
    seeds = list(map(int, contract["confirmation_seeds"]))
    if seeds != list(range(3900, 3920)) or len(set(seeds)) != 20:
        raise RuntimeError("confirmation seed cohort changed")
    if set(seeds) & set(contract["prior_development_seeds"]):
        raise RuntimeError("confirmation seeds overlap prior development")
    if contract.get("confirmation_seeds_opened") is not False:
        raise RuntimeError("contract must be frozen before opening seeds")
    if int(contract["workers"]) * int(contract["threads_per_worker"]) > 16:
        raise RuntimeError("CPU worker contract exceeds logical CPU budget")
    for field in ("source_checkpoint", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact checksum mismatch: {field}")
    trace = contract["external_trace"]
    if sha256(trace["path"]) != trace["sha256"]:
        raise RuntimeError("external trace checksum mismatch")
    if contract["decision_rules"] != {
        "no_retuning_after_open": True,
        "no_candidate_selection_on_confirmation": True,
        "failed_claims_are_reported_not_repaired": True,
        "universal_lifetime_superiority_not_claimed": True,
    }:
        raise RuntimeError("post-confirmation decision rules changed")
    return contract


def evaluate_task(contract_path: str, scenario: dict, policy: str, seed: int, smoke: bool):
    contract = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), int(seed) + len(policy))
    runtime = dict(contract)
    runtime["development_seeds"] = [int(seed)]
    runtime["observation_schema"] = STEP3_CH_CONTEXT_SCHEMA
    if smoke:
        runtime["horizon"] = 40
    trace = load_trace(resolve(contract["external_trace"]["path"])) if scenario.get("use_external_trace") else None
    environments = build_transfer_environments(runtime, scenario, trace=trace)
    qos = Step3QoSConstraintConfig.from_payload(json.loads(resolve(contract["qos_config"]).read_text()))
    agent = load_agent(resolve(contract["source_checkpoint"]))[0] if policy == "hta_mac" else None
    rows = []
    for env in environments:
        observation, mask, _ = env.reset()
        dual = OnlinePrimalDualQoS(qos, contract["primal_dual"], int(contract["budget"]))
        done = False
        consumed = 0.0
        allocated = 0
        while not done:
            state, active, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
            if policy == "hta_mac":
                action, _ = agent.act(state, active, epsilon=0.0, caps=caps, budget=int(contract["budget"]))
            elif policy == "energy_proportional":
                action = energy_proportional_action(env, active, int(contract["budget"]), float(contract["energy_proportional_score_exponent"]))
            elif policy == "online_primal_dual":
                action = dual.action(env, active, caps)
            else:
                raise ValueError(policy)
            allocated += int(action.sum())
            observation, mask, done, info = env.step(action)
            consumed += float(np.asarray(info["energy_trace"]["consumed"]).sum())
        counts = env.step3_qos_counts
        demand = max(1, int(counts["demand"]))
        event = env.base.t_fnd is not None
        rows.append({
            "policy": policy, "seed": int(seed), "target_rank": int(env.target_rank),
            "delivery_ratio": int(counts["delivered"]) / demand,
            "stale_ratio": int(counts["stale"]) / demand,
            "fairness": float(counts["episode_service_fairness"]),
            "joint_qos_pass": bool(
                int(counts["delivered"]) / demand >= qos.minimum_delivery_ratio
                and int(counts["stale"]) / demand <= qos.maximum_stale_drop_ratio
                and float(counts["episode_service_fairness"]) >= qos.minimum_queue_fairness
            ),
            "fnd_event_observed": bool(event),
            "restricted_survival_rounds": int(env.base.t_fnd if event else runtime["horizon"]),
            "global_packets": int(env.base.total_packets),
            "network_energy_j": consumed,
            "packets_per_j": int(env.base.total_packets) / max(consumed, 1e-12),
            "allocated_slots": allocated,
        })
    return scenario["id"], policy, int(seed), rows


def seed_summary(rows: list[dict]) -> dict:
    metrics = ("delivery_ratio", "stale_ratio", "fairness", "restricted_survival_rounds", "global_packets", "network_energy_j", "packets_per_j", "allocated_slots")
    return {
        **{metric: float(np.mean([row[metric] for row in rows])) for metric in metrics},
        "joint_qos_pass_rate": float(np.mean([row["joint_qos_pass"] for row in rows])),
        "fnd_event_count": int(sum(row["fnd_event_observed"] for row in rows)),
        "rank_units": len(rows),
    }


def paired(differences: np.ndarray, *, alternative: str, rng, resamples: int) -> dict:
    differences = np.asarray(differences, dtype=np.float64)
    draws = rng.choice(differences, size=(int(resamples), len(differences)), replace=True).mean(axis=1)
    if np.allclose(differences, 0.0):
        two_sided = one_sided = 1.0
    else:
        two_sided = float(wilcoxon(differences, alternative="two-sided", method="auto").pvalue)
        one_sided = float(wilcoxon(differences, alternative=alternative, method="auto").pvalue)
    std = float(differences.std(ddof=1))
    dz = float(differences.mean() / std) if std > 0.0 else (math.inf if differences.mean() > 0 else -math.inf if differences.mean() < 0 else 0.0)
    correction = 1.0 - 3.0 / max(1.0, 4.0 * len(differences) - 5.0)
    return {
        "n_seed_pairs": len(differences), "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "bootstrap_95_ci": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "wilcoxon_two_sided_p": two_sided, "wilcoxon_directional_p": one_sided,
        "paired_hedges_g": float(dz * correction) if np.isfinite(dz) else str(dz),
    }


def holm_adjust(entries: dict[str, float]) -> dict[str, float]:
    ordered = sorted(entries, key=entries.get)
    adjusted, running = {}, 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * float(entries[key])))
        adjusted[key] = running
    return adjusted


def analyze(contract: dict, raw: dict) -> tuple[dict, dict]:
    inference = contract["inference"]
    rng = np.random.default_rng(int(inference["bootstrap_seed"]))
    results, delivery_family = {}, {}
    for scenario in contract["scenarios"]:
        identifier = scenario["id"]
        results[identifier] = {"policies": {}, "comparisons": {}}
        for policy in contract["policies"]:
            summaries = {str(seed): seed_summary(raw[identifier][policy][str(seed)]) for seed in contract["confirmation_seeds"]}
            results[identifier]["policies"][policy] = {
                "seed_summaries": summaries,
                "mean": {key: float(np.mean([row[key] for row in summaries.values()])) for key in next(iter(summaries.values())) if key not in {"rank_units"}},
            }
        hta = results[identifier]["policies"]["hta_mac"]["seed_summaries"]
        for baseline in ("energy_proportional", "online_primal_dual"):
            base = results[identifier]["policies"][baseline]["seed_summaries"]
            comparisons = {}
            for metric, alternative in (("delivery_ratio", "greater"), ("restricted_survival_rounds", "greater"), ("packets_per_j", "greater"), ("stale_ratio", "less"), ("fairness", "greater")):
                multiplier = -1.0 if alternative == "less" else 1.0
                values = np.asarray([multiplier * (hta[str(seed)][metric] - base[str(seed)][metric]) for seed in contract["confirmation_seeds"]])
                item = paired(values, alternative="greater", rng=rng, resamples=inference["bootstrap_resamples"])
                item["reported_difference"] = "baseline_minus_hta" if alternative == "less" else "hta_minus_baseline"
                comparisons[metric] = item
            relative_efficiency = np.asarray([(hta[str(seed)]["packets_per_j"] - base[str(seed)]["packets_per_j"]) / max(base[str(seed)]["packets_per_j"], 1e-12) for seed in contract["confirmation_seeds"]])
            comparisons["packets_per_j_relative"] = paired(relative_efficiency, alternative="greater", rng=rng, resamples=inference["bootstrap_resamples"])
            results[identifier]["comparisons"][baseline] = comparisons
            if baseline == "energy_proportional":
                delivery_family[identifier] = comparisons["delivery_ratio"]["wilcoxon_directional_p"]
    adjusted = holm_adjust(delivery_family)
    for scenario, value in adjusted.items():
        results[scenario]["comparisons"]["energy_proportional"]["delivery_ratio"]["holm_adjusted_directional_p"] = value

    alpha = float(inference["alpha"])
    ref = results["reference_100"]["comparisons"]
    claims = contract["predeclared_claims"]
    reference_delivery = ref["energy_proportional"]["delivery_ratio"]
    robustness = {
        scenario: bool(
            results[scenario]["comparisons"]["energy_proportional"]["delivery_ratio"]["mean_difference"] > 0.0
            and results[scenario]["comparisons"]["energy_proportional"]["delivery_ratio"]["holm_adjusted_directional_p"] < alpha
        ) for scenario in claims["delivery_robustness_vs_energy_family"]["scenarios"]
    }
    nodes_life = results["nodes_20"]["comparisons"]["energy_proportional"]["restricted_survival_rounds"]
    ni = claims["reference_noninferiority_vs_primal_dual"]
    primal = ref["online_primal_dual"]
    noninferiority = {
        "delivery": primal["delivery_ratio"]["bootstrap_95_ci"][0] > -float(ni["delivery_margin_absolute"]),
        "rmst": primal["restricted_survival_rounds"]["bootstrap_95_ci"][0] > -float(ni["rmst_margin_rounds"]),
        "packets_per_j": primal["packets_per_j_relative"]["bootstrap_95_ci"][0] > -float(ni["packets_per_j_margin_fraction"]),
    }
    decisions = {
        "reference_delivery_superiority_vs_energy": bool(reference_delivery["mean_difference"] > 0.0 and reference_delivery["wilcoxon_directional_p"] < alpha and reference_delivery["bootstrap_95_ci"][0] > 0.0),
        "delivery_robustness_vs_energy_by_scenario": robustness,
        "delivery_robustness_all_scenarios": all(robustness.values()),
        "nodes20_lifetime_superiority_vs_energy": bool(nodes_life["mean_difference"] > 0.0 and nodes_life["wilcoxon_directional_p"] < alpha and nodes_life["bootstrap_95_ci"][0] > 0.0),
        "reference_noninferiority_vs_primal_dual_components": noninferiority,
        "reference_noninferiority_vs_primal_dual": all(noninferiority.values()),
        "universal_lifetime_superiority_claimed": False,
        "retuning_authorized": False,
    }
    return results, decisions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    contract_path = resolve(args.contract)
    output_path = resolve(args.output)
    contract = load_contract(contract_path)
    seeds = [int(contract["smoke_seed"])] if args.smoke else list(map(int, contract["confirmation_seeds"]))
    scenarios = contract["scenarios"][:2] if args.smoke else contract["scenarios"]
    started = time.perf_counter()
    raw = {row["id"]: {policy: {} for policy in contract["policies"]} for row in scenarios}
    tasks = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
        for scenario in scenarios:
            for policy in contract["policies"]:
                for seed in seeds:
                    tasks.append(pool.submit(evaluate_task, str(contract_path), scenario, policy, seed, bool(args.smoke)))
        completed = 0
        for future in concurrent.futures.as_completed(tasks):
            scenario, policy, seed, rows = future.result()
            raw[scenario][policy][str(seed)] = rows
            completed += 1
            if completed % max(1, len(tasks) // 20) == 0 or completed == len(tasks):
                print(f"CONFIRMATION_PROGRESS={completed}/{len(tasks)}", flush=True)
    checks = {
        "all_tasks_complete": all(len(raw[row["id"]][policy]) == len(seeds) for row in scenarios for policy in contract["policies"]),
        "paired_seed_cohorts": all(set(raw[row["id"]][policy]) == set(map(str, seeds)) for row in scenarios for policy in contract["policies"]),
        "seed_scope_valid": bool(
            (args.smoke and seeds == [int(contract["smoke_seed"])])
            or (not args.smoke and seeds == list(map(int, contract["confirmation_seeds"])))
        ),
        "no_training_or_retuning": True,
    }
    if args.smoke:
        results, decisions = {}, {"smoke_only": True}
    else:
        results, decisions = analyze(contract, raw)
    payload = {
        "schema_version": 1,
        "status": "preconfirmation_smoke_pass" if args.smoke and all(checks.values()) else "final_confirmation_complete" if not args.smoke and all(checks.values()) else "confirmation_incomplete",
        "smoke": bool(args.smoke), "contract": str(contract_path), "contract_sha256": sha256(contract_path),
        "runner_sha256": sha256(Path(__file__)), "source_checkpoint_sha256": sha256(contract["source_checkpoint"]),
        "confirmation_seeds": [] if args.smoke else seeds,
        "confirmation_seeds_opened": not args.smoke,
        "raw": raw, "results": results, "decisions": decisions, "checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
        "selection_or_retuning_performed": False,
        "claim_boundary": contract["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "decisions": decisions, "elapsed_seconds": payload["elapsed_seconds"], "output": str(output_path)}, indent=2), flush=True)
    return 0 if payload["status"] in {"preconfirmation_smoke_pass", "final_confirmation_complete"} else 3


if __name__ == "__main__":
    raise SystemExit(main())
