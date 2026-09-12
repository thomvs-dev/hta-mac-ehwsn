"""Develop and internally validate raw-policy-relative residual retention.

The controller keeps more of the frozen C51 action when the raw policy itself
predicts that substantially more demand can be served.  An observable CH-energy
gate fades this extra retention as the scheduled CH approaches the frozen
reserve floor.  This avoids using hidden scenario arrival-rate parameters.
"""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ch_depletion_risk import validate_ch_risk_config
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from experiments.diagnose_step3_delivery_feasibility import build_environments, load_agent
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.evaluate_step3_final_matched_baselines import load_ranker
from experiments.evaluate_step4_publication_evidence import (
    aggregate,
    build_transfer_environments,
    evaluate_rows,
    load_trace,
)
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


def expanded_prohibited(contract: dict) -> set[int]:
    result = set()
    for lower, upper in contract["prohibited_prior_seed_ranges"]:
        result.update(range(int(lower), int(upper) + 1))
    return result


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("parent_contract"):
        parent_path = resolve(contract["parent_contract"])
        if sha256(parent_path) != contract["parent_contract_sha256"]:
            raise RuntimeError("parent contract hash mismatch")
        parent = json.loads(parent_path.read_text(encoding="utf-8"))
        overrides = {
            key: value for key, value in contract.items()
            if key not in {"parent_contract", "parent_contract_sha256"}
        }
        contract = {**parent, **overrides}
    if contract.get("status") != "frozen_before_step4_load_adaptive_development":
        raise RuntimeError("load-adaptive development contract is not frozen")
    for field in (
        "source_checkpoint", "selected_ranker_checkpoint", "qos_config",
        "risk_config", "external_trace",
    ):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {field}")
    calibration = set(contract["calibration_seeds"])
    validation = set(contract["validation_seeds"])
    confirmation = set(contract["confirmation_seeds"])
    prohibited = expanded_prohibited(contract)
    if calibration & (validation | confirmation | prohibited):
        raise RuntimeError("calibration cohort is not independent")
    if validation & (confirmation | prohibited):
        raise RuntimeError("validation cohort is not independent")
    if contract.get("confirmation_seeds_opened") is not False:
        raise RuntimeError("confirmation seeds must remain sealed")
    if int(contract["parallel_workers"]) * int(contract["threads_per_worker"]) > 16:
        raise RuntimeError("CPU contract exceeds 16 logical threads")
    if len(contract["candidates"]) != 3:
        raise RuntimeError("exactly three frozen candidates are required")
    return contract


def adaptive_upper_target(env, raw_action, candidate: dict, band: dict) -> dict:
    offered = int(env.base.queue[env.members][env.base.alive[env.members]].sum())
    # Use the frozen policy's current service opportunity, not actual cumulative
    # delivery.  Cumulative delivery already contains residual throttling and
    # would otherwise feed the old ~0.30 cap back into the adaptive target.
    raw_ratio = float(np.clip(int(np.asarray(raw_action).sum()) / max(1, offered), 0.0, 1.0))
    base_upper = float(band["upper_delivery_target"])
    retained_target = min(
        float(candidate["maximum_upper_target"]),
        float(candidate["raw_policy_retention"]) * raw_ratio,
    )
    ch_energy_ratio = float(np.clip(
        env.base.energy[int(env.ch)] / env.base.cfg.initial_energy_j, 0.0, 1.0
    ))
    reserve = float(band["reserve_floor"])
    normalized_energy = float(np.clip((ch_energy_ratio - reserve) / max(1e-12, 1.0 - reserve), 0.0, 1.0))
    minimum_gate = float(candidate.get("minimum_energy_gate", 0.0))
    if not 0.0 <= minimum_gate <= 1.0:
        raise ValueError("minimum_energy_gate must be in [0, 1]")
    energy_gate = minimum_gate + (1.0 - minimum_gate) * (
        normalized_energy ** float(candidate["energy_gate_exponent"])
    )
    upper = base_upper + energy_gate * max(0.0, retained_target - base_upper)
    return {
        "upper_target": float(np.clip(upper, base_upper, candidate["maximum_upper_target"])),
        "raw_instant_service_ratio": raw_ratio,
        "ch_energy_ratio": ch_energy_ratio,
        "energy_gate": energy_gate,
    }


def adaptive_removal_count(action, raw_action, env, band, candidate) -> tuple[int, dict]:
    audit = adaptive_upper_target(env, raw_action, candidate, band)
    counts = env.step3_qos_counts
    offered = int(env.base.queue[env.members][env.base.alive[env.members]].sum())
    predicted_demand = int(counts["demand"]) + offered
    lower_required = int(math.ceil(float(band["lower_delivery_target"]) * predicted_demand))
    removal_upper = (
        float(band["upper_delivery_target"])
        if candidate.get("mode") == "load_conditioned_removal_bypass"
        else audit["upper_target"]
    )
    upper_allowed = int(math.floor(removal_upper * predicted_demand))
    predicted_delivery = int(counts["delivered"]) + int(np.asarray(action).sum())
    frozen_count = min(
        max(0, predicted_delivery - upper_allowed),
        max(0, predicted_delivery - lower_required),
    )
    count = frozen_count
    if candidate.get("mode") == "load_conditioned_removal_bypass":
        raw_ratio = audit["raw_instant_service_ratio"]
        bypass = float(np.clip(
            float(candidate["bypass_intercept"])
            + float(candidate["bypass_slope"]) * (raw_ratio - 0.5),
            float(candidate["minimum_bypass"]),
            float(candidate["maximum_bypass"]),
        ))
        ch_energy_ratio = audit["ch_energy_ratio"]
        reserve = float(band["reserve_floor"])
        normalized_energy = float(np.clip(
            (ch_energy_ratio - reserve) / max(1e-12, 1.0 - reserve), 0.0, 1.0
        ))
        minimum_energy_factor = float(candidate["minimum_energy_factor"])
        bypass *= minimum_energy_factor + (1.0 - minimum_energy_factor) * normalized_energy
        # The cumulative excess may be orders of magnitude larger than this
        # round's action.  Interpolate over slots that can actually be removed;
        # otherwise any bypass below 1.0 still strips the entire current action.
        currently_removable = min(frozen_count, int(np.asarray(action).sum()))
        count = int(math.floor(currently_removable * (1.0 - bypass) + 1e-12))
        audit["removal_bypass"] = bypass
        audit["currently_removable_slots"] = currently_removable
    audit.update({
        "predicted_demand": predicted_demand,
        "frozen_removed_slots": int(frozen_count),
        "removed_slots": int(count),
    })
    return int(count), audit


def evaluate_adaptive(contract: dict, environments, candidate: dict) -> dict:
    qos = Step3QoSConstraintConfig.from_payload(json.loads(resolve(contract["qos_config"]).read_text()))
    agent, _ = load_agent(resolve(contract["source_checkpoint"]))
    ranker = load_ranker(resolve(contract["selected_ranker_checkpoint"]))
    rows, audits = [], []
    runtime = {**contract, "band": contract["band"]}
    for env in environments:
        observation, mask, _ = env.reset()
        done = False
        energy = 0.0
        allocated_slots = 0
        while not done:
            _, padded_mask, caps, raw, q_values, added = base_and_added(
                agent, env, observation, mask, runtime
            )
            if candidate.get("mode") == "energy_threshold_raw_switch":
                ch_energy_ratio = float(np.clip(
                    env.base.energy[int(env.ch)] / env.base.cfg.initial_energy_j, 0.0, 1.0
                ))
                raw_selected = ch_energy_ratio >= float(candidate["raw_until_energy_ratio"])
                if raw_selected:
                    action = np.asarray(raw, dtype=np.int64).copy()
                    count = 0
                else:
                    count = removal_count(added, env, contract["band"])
                    action = apply_ranker(
                        ranker, env, q_values, added, caps, padded_mask, count
                    )
                audit = {
                    "upper_target": float(contract["band"]["upper_delivery_target"]),
                    "raw_instant_service_ratio": float(np.clip(
                        int(np.asarray(raw).sum()) /
                        max(1, int(env.base.queue[env.members][env.base.alive[env.members]].sum())),
                        0.0, 1.0,
                    )),
                    "ch_energy_ratio": ch_energy_ratio,
                    "energy_gate": float(raw_selected),
                    "predicted_demand": int(env.step3_qos_counts["demand"]),
                    "frozen_removed_slots": int(count),
                    "removed_slots": int(count),
                    "raw_switch_selected": int(raw_selected),
                }
            else:
                count, audit = adaptive_removal_count(
                    added, raw, env, contract["band"], candidate
                )
                action = apply_ranker(ranker, env, q_values, added, caps, padded_mask, count)
            audits.append(audit)
            allocated_slots += int(action.sum())
            observation, mask, done, info = env.step(action)
            energy += float(np.asarray(info["energy_trace"]["consumed"]).sum())
        counts = env.step3_qos_counts
        demand = max(1, int(counts["demand"]))
        delivery = int(counts["delivered"]) / demand
        stale = int(counts["stale"]) / demand
        fairness = float(counts["episode_service_fairness"])
        event = env.base.t_fnd is not None
        fnd = int(env.base.t_fnd if event else contract["horizon"])
        rows.append({
            "seed": int(env.seed), "target_rank": int(env.target_rank),
            "delivery_ratio": delivery, "stale_ratio": stale,
            "episode_service_fairness": fairness,
            "joint_qos_pass": bool(
                delivery >= qos.minimum_delivery_ratio
                and stale <= qos.maximum_stale_drop_ratio
                and fairness >= qos.minimum_queue_fairness
            ),
            "fnd_event_observed": bool(event),
            "fnd_or_censoring_round": fnd,
            "restricted_survival_rounds": fnd,
            "global_packets_delivered": int(env.base.total_packets),
            "network_energy_j": energy,
            "global_packets_per_j": int(env.base.total_packets) / max(energy, 1e-12),
            "allocated_slots": allocated_slots,
        })
    result = aggregate(rows)
    result["audit"] = {
        "steps": len(audits),
        "mean_upper_target": float(np.mean([a["upper_target"] for a in audits])),
        "mean_raw_instant_service_ratio": float(np.mean([a["raw_instant_service_ratio"] for a in audits])),
        "mean_energy_gate": float(np.mean([a["energy_gate"] for a in audits])),
        "removed_slots": int(sum(a["removed_slots"] for a in audits)),
        "frozen_removed_slots": int(sum(a["frozen_removed_slots"] for a in audits)),
        "mean_removal_bypass": float(np.mean([a.get("removal_bypass", 0.0) for a in audits])),
        "raw_switch_fraction": float(np.mean([a.get("raw_switch_selected", 0.0) for a in audits])),
    }
    result["detail_rows"] = rows
    return result


def build_suite(contract: dict, seeds: list[int]) -> dict:
    runtime = dict(contract)
    runtime["development_seeds"] = list(seeds)
    risk = validate_ch_risk_config(json.loads(resolve(contract["risk_config"]).read_text()))
    primary, _, _ = build_environments(None, risk, int(contract["horizon"]), seeds=seeds)
    suite = {"primary": primary}
    trace = load_trace(resolve(contract["external_trace"]))
    for scenario in contract["transfer_scenarios"]:
        suite[scenario["id"]] = build_transfer_environments(
            runtime, scenario, trace=trace if scenario.get("use_external_trace") else None
        )
    return suite


def baseline_worker(contract_path: str, seeds: list[int]) -> dict:
    contract = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), 20260816)
    suite = build_suite(contract, seeds)
    result = {}
    for name, envs in suite.items():
        result[name] = {
            "frozen_residual": aggregate(evaluate_rows(contract, envs, "learned_listwise_residual", contract["band"])),
            "primal_dual": aggregate(evaluate_rows(contract, envs, "online_primal_dual_qos", contract["band"])),
        }
    return result


def candidate_worker(contract_path: str, seeds: list[int], candidate: dict) -> dict:
    contract = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), 20260816 + len(candidate["id"]))
    suite = build_suite(contract, seeds)
    return {
        "candidate": candidate,
        "scenarios": {name: evaluate_adaptive(contract, envs, candidate) for name, envs in suite.items()},
    }


def gate_candidate(contract: dict, row: dict, baselines: dict) -> dict:
    gates = contract["calibration_gates"]
    scenarios = row["scenarios"]
    frozen = {name: values["frozen_residual"] for name, values in baselines.items()}
    fnd_deltas = {
        name: scenarios[name]["mean_restricted_survival_rounds"] - frozen[name]["mean_restricted_survival_rounds"]
        for name in scenarios
    }
    checks = {
        "primary_joint_qos": scenarios["primary"]["joint_qos_pass_count"] >= int(gates["minimum_primary_joint_qos_pairs"]),
        "reference_delivery": scenarios["reference_100"]["mean_delivery_ratio"] - frozen["reference_100"]["mean_delivery_ratio"] >= float(gates["minimum_reference_delivery_improvement"]),
        "low_traffic_delivery": scenarios["traffic_low"]["mean_delivery_ratio"] >= float(gates["minimum_low_traffic_delivery"]),
        "nodes20_delivery": scenarios["nodes_20"]["mean_delivery_ratio"] >= float(gates["minimum_nodes20_delivery"]),
        "primary_fnd": fnd_deltas["primary"] >= -float(gates["maximum_primary_fnd_degradation_rounds"]),
        "transfer_fnd": min(v for k, v in fnd_deltas.items() if k != "primary") >= -float(gates["maximum_any_transfer_fnd_degradation_rounds"]),
        "reference_efficiency": scenarios["reference_100"]["mean_global_packets_per_j"] >= frozen["reference_100"]["mean_global_packets_per_j"] * (1.0 - float(gates["maximum_reference_efficiency_degradation_fraction"])),
    }
    delivery_improvements = {
        name: scenarios[name]["mean_delivery_ratio"] - frozen[name]["mean_delivery_ratio"]
        for name in scenarios
    }
    return {
        **row, "fnd_deltas_vs_frozen": fnd_deltas,
        "delivery_improvements_vs_frozen": delivery_improvements,
        "checks": checks, "gate_pass": all(checks.values()),
    }


def run_phase(contract_path: Path, contract: dict, seeds: list[int], candidates: list[dict]):
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["parallel_workers"])) as pool:
        baseline_future = pool.submit(baseline_worker, str(contract_path), seeds)
        candidate_futures = [
            pool.submit(candidate_worker, str(contract_path), seeds, candidate)
            for candidate in candidates
        ]
        baselines = baseline_future.result()
        rows = [gate_candidate(contract, future.result(), baselines) for future in candidate_futures]
    return baselines, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    contract_path = resolve(args.contract)
    output_path = resolve(args.output)
    contract = load_contract(contract_path)
    if args.smoke:
        contract = dict(contract)
        contract["calibration_seeds"] = contract["calibration_seeds"][:1]
        contract["validation_seeds"] = contract["validation_seeds"][:1]
        contract["horizon"] = 40
        contract["calibration_gates"] = {
            **contract["calibration_gates"],
            "minimum_primary_joint_qos_pairs": 0,
            "minimum_reference_delivery_improvement": -1.0,
            "minimum_low_traffic_delivery": 0.0,
            "minimum_nodes20_delivery": 0.0,
            "maximum_primary_fnd_degradation_rounds": 100.0,
            "maximum_any_transfer_fnd_degradation_rounds": 100.0,
            "maximum_reference_efficiency_degradation_fraction": 1.0,
        }
        runtime_path = output_path.parent / "smoke_runtime_contract.json"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
        contract_path = runtime_path
    started = time.perf_counter()
    calibration_baselines, calibration = run_phase(
        contract_path, contract, contract["calibration_seeds"], contract["candidates"]
    )
    passing = [row for row in calibration if row["gate_pass"]]
    selected = max(
        passing,
        key=lambda row: (
            min(row["delivery_improvements_vs_frozen"].values()),
            float(np.mean(list(row["fnd_deltas_vs_frozen"].values()))),
            -next(i for i, c in enumerate(contract["candidates"]) if c["id"] == row["candidate"]["id"]),
        ),
    ) if passing else None
    validation_baselines = validation = None
    if selected is not None:
        validation_baselines, validation_rows = run_phase(
            contract_path, contract, contract["validation_seeds"], [selected["candidate"]]
        )
        validation = validation_rows[0]
    payload = {
        "schema_version": 1,
        "status": "load_adaptive_internal_validation_complete" if validation else "no_load_adaptive_candidate_passed",
        "smoke": bool(args.smoke),
        "contract_sha256": sha256(contract_path),
        "evaluator_sha256": sha256(Path(__file__)),
        "calibration_seeds": contract["calibration_seeds"],
        "validation_seeds": contract["validation_seeds"],
        "confirmation_seeds_opened": False,
        "calibration_baselines": calibration_baselines,
        "calibration_candidates": calibration,
        "selected_candidate": selected["candidate"] if selected else None,
        "validation_baselines": validation_baselines,
        "validation": validation,
        "publication_evidence": False,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": contract["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "selected": payload["selected_candidate"],
        "elapsed_seconds": payload["elapsed_seconds"], "output": str(output_path),
    }, indent=2), flush=True)
    return 0 if validation else 3


if __name__ == "__main__":
    raise SystemExit(main())
