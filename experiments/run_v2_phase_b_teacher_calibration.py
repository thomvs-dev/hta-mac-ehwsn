"""Calibrate and freeze a deterministic marginal-utility teacher."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.marginal_utility_teacher import MarginalTeacherConfig, marginal_utility_action
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.evaluate_step4_publication_evidence import build_transfer_environments


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(value: str | Path) -> str:
    return hashlib.sha256(resolve(value).read_bytes()).hexdigest()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_v2_phase_b_teacher_calibration":
        raise RuntimeError("Phase B calibration contract is not frozen")
    if contract.get("phase") != "B_calibration_only_no_training":
        raise RuntimeError("only deterministic teacher calibration is authorized")
    for field in ("source_phase_a2_results", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact checksum mismatch: {field}")
    phase_a2 = json.loads(resolve(contract["source_phase_a2_results"]).read_text(encoding="utf-8"))
    if phase_a2["gate_a"]["gate_a_pass"] is not True:
        raise RuntimeError("Gate A must pass before Phase B")
    calibration = set(contract["calibration_seeds"])
    gate = set(contract["reserved_gate_seeds"])
    prohibited = set(contract["permanently_opened_confirmation_seeds"]) | set(contract["all_prior_v2_seeds"])
    if calibration & gate or calibration & prohibited or gate & prohibited:
        raise RuntimeError("Phase B seed cohorts overlap")
    if int(contract["workers"]) * int(contract["threads_per_worker"]) > 8:
        raise RuntimeError("worker contract exceeds stable host capacity")
    return contract


def expand_candidates(contract: dict) -> list[dict]:
    grid, fixed = contract["candidate_grid"], contract["candidate_grid"]["fixed"]
    candidates = []
    for index, (energy, stale, fairness) in enumerate(itertools.product(
        grid["energy_weight"], grid["stale_weight"], grid["fairness_weight"]
    ), start=1):
        candidates.append({
            "candidate_id": f"mu_{index:02d}_e{energy:g}_s{stale:g}_f{fairness:g}",
            "energy_weight": float(energy), "stale_weight": float(stale),
            "fairness_weight": float(fairness), **{key: float(value) for key, value in fixed.items()},
        })
    if len(candidates) != 24:
        raise RuntimeError("frozen calibration grid must contain 24 candidates")
    return candidates


def evaluate_candidate(contract_path: str, candidate: dict) -> dict:
    contract = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), 121_000 + int(candidate["candidate_id"].split("_")[1]))
    runtime = dict(contract)
    runtime["development_seeds"] = list(contract["calibration_seeds"])
    runtime["observation_schema"] = STEP3_CH_CONTEXT_SCHEMA
    config = MarginalTeacherConfig.from_payload(candidate)
    envs = build_transfer_environments(runtime, contract["scenario"], trace=None)
    rows = []
    for env in envs:
        observation, mask, _ = env.reset()
        done = False; energy = 0.0; allocated = 0; violations = 0
        while not done:
            _, active, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
            action = marginal_utility_action(env, active, caps, int(contract["budget"]), config)
            violations += int(
                int(action.sum()) > int(contract["budget"])
                or np.any(action < 0) or np.any(action > caps) or np.any(action[~active] != 0)
            )
            allocated += int(action.sum())
            observation, mask, done, info = env.step(action)
            role = info["energy_trace"]["role_energy"]
            reconstructed = sum(float(np.asarray(role[name]).sum()) for name in ("member_tx", "ch_rx", "ch_aggregate", "ch_tx_bs", "idle"))
            consumed = float(np.asarray(info["energy_trace"]["consumed"]).sum())
            violations += int(abs(reconstructed - consumed) > 1e-12)
            energy += consumed
        counts = env.step3_qos_counts; demand = max(1, int(counts["demand"]))
        rows.append({
            "seed": int(env.seed), "target_rank": int(env.target_rank),
            "delivery_ratio": int(counts["delivered"]) / demand,
            "packets_per_j": int(env.base.total_packets) / max(energy, 1e-12),
            "stale_ratio": int(counts["stale"]) / demand,
            "fairness": float(counts["episode_service_fairness"]),
            "rmst": int(env.base.t_fnd if env.base.t_fnd is not None else contract["horizon"]),
            "allocated_slots": allocated, "feasibility_violations": violations,
        })
    seed_summaries = {}
    for seed in contract["calibration_seeds"]:
        selected = [row for row in rows if row["seed"] == int(seed)]
        seed_summaries[str(seed)] = {
            key: float(np.mean([row[key] for row in selected]))
            for key in ("delivery_ratio", "packets_per_j", "stale_ratio", "fairness", "rmst", "allocated_slots")
        } | {"feasibility_violations": int(sum(row["feasibility_violations"] for row in selected))}
    aggregate = {
        f"mean_{key}": float(np.mean([row[key] for row in seed_summaries.values()]))
        for key in ("delivery_ratio", "packets_per_j", "stale_ratio", "fairness", "rmst", "allocated_slots")
    }
    aggregate["feasibility_violations"] = int(sum(row["feasibility_violations"] for row in seed_summaries.values()))
    return {"candidate": candidate, "aggregate": aggregate, "seed_summaries": seed_summaries}


def gate_checks(aggregate: dict, targets: dict) -> dict:
    return {
        "delivery": aggregate["mean_delivery_ratio"] >= float(targets["delivery_min"]),
        "packets_per_j": aggregate["mean_packets_per_j"] >= float(targets["packets_per_j_min"]),
        "stale_ratio": aggregate["mean_stale_ratio"] <= float(targets["stale_ratio_max"]),
        "fairness": aggregate["mean_fairness"] >= float(targets["fairness_min"]),
        "rmst": aggregate["mean_rmst"] >= float(targets["rmst_min"]),
        "feasibility": aggregate["feasibility_violations"] <= int(targets["feasibility_violations_max"]),
    }


def normalized_violation(aggregate: dict, targets: dict) -> float:
    terms = (
        max(0.0, float(targets["delivery_min"]) - aggregate["mean_delivery_ratio"]) / float(targets["delivery_min"]),
        max(0.0, float(targets["packets_per_j_min"]) - aggregate["mean_packets_per_j"]) / float(targets["packets_per_j_min"]),
        max(0.0, aggregate["mean_stale_ratio"] - float(targets["stale_ratio_max"])) / float(targets["stale_ratio_max"]),
        max(0.0, float(targets["fairness_min"]) - aggregate["mean_fairness"]) / float(targets["fairness_min"]),
        max(0.0, float(targets["rmst_min"]) - aggregate["mean_rmst"]) / float(targets["rmst_min"]),
        float(max(0, aggregate["feasibility_violations"])),
    )
    return float(sum(terms))


def select_candidate(results: list[dict], targets: dict) -> tuple[dict, bool]:
    for row in results:
        row["checks"] = gate_checks(row["aggregate"], targets)
        row["passes_all_calibration_targets"] = all(row["checks"].values())
        row["normalized_total_violation"] = normalized_violation(row["aggregate"], targets)
    passing = [row for row in results if row["passes_all_calibration_targets"]]
    if passing:
        return min(passing, key=lambda row: (
            -row["aggregate"]["mean_packets_per_j"],
            -row["aggregate"]["mean_delivery_ratio"], row["candidate"]["candidate_id"],
        )), True
    return min(results, key=lambda row: (
        row["normalized_total_violation"], -row["aggregate"]["mean_packets_per_j"],
        -row["aggregate"]["mean_delivery_ratio"], row["candidate"]["candidate_id"],
    )), False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract_path, output_dir = resolve(args.contract), resolve(args.output_dir)
    contract = load_contract(contract_path); candidates = expand_candidates(contract)
    output_dir.mkdir(parents=True, exist_ok=True); started = time.perf_counter()
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
        futures = [pool.submit(evaluate_candidate, str(contract_path), candidate) for candidate in candidates]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            print(f"PHASE_B_CALIBRATION_PROGRESS={index}/{len(futures)}", flush=True)
    selected, passed = select_candidate(results, contract["gate_b_targets"])
    results.sort(key=lambda row: row["candidate"]["candidate_id"])
    selected_payload = {
        "schema_version": 1, "status": "teacher_frozen_after_calibration",
        "candidate": selected["candidate"], "calibration_aggregate": selected["aggregate"],
        "calibration_checks": selected["checks"], "calibration_passed_all_targets": passed,
        "normalized_total_violation": selected["normalized_total_violation"],
        "selection_rule": contract["selection_rule"],
        "calibration_seeds": contract["calibration_seeds"],
        "reserved_gate_seeds_opened": False,
        "teacher_implementation": "agents/marginal_utility_teacher.py",
        "teacher_implementation_sha256": sha256("agents/marginal_utility_teacher.py"),
        "no_training": True,
    }
    selected_path = output_dir / "selected_teacher.json"
    selected_path.write_text(json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8")
    payload = {
        "schema_version": 1, "status": "phase_b_calibration_complete",
        "contract_sha256": sha256(contract_path), "runner_sha256": sha256(Path(__file__)),
        "calibration_seeds": contract["calibration_seeds"], "reserved_gate_seeds_opened": False,
        "candidate_count": len(results), "candidates": results,
        "selected_candidate_id": selected["candidate"]["candidate_id"],
        "selected_teacher": str(selected_path), "selected_teacher_sha256": sha256(selected_path),
        "selected_passed_all_calibration_targets": passed,
        "elapsed_seconds": time.perf_counter() - started, "neural_training_performed": False,
    }
    (output_dir / "calibration_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "selected": payload["selected_candidate_id"],
        "calibration_passed": passed, "aggregate": selected["aggregate"],
        "selected_teacher": str(selected_path), "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
