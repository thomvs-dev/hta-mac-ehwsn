"""Calibrate and freeze the Phase B2 target-aware deterministic teacher."""

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
from agents.guarded_marginal_teacher import GuardedTeacherConfig, guarded_marginal_action
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.evaluate_step4_publication_evidence import build_transfer_environments
from experiments.run_v2_phase_b_teacher_calibration import gate_checks, select_candidate


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(value: str | Path) -> str:
    return hashlib.sha256(resolve(value).read_bytes()).hexdigest()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_v2_phase_b2_guarded_teacher_calibration":
        raise RuntimeError("Phase B2 calibration contract is not frozen")
    if contract.get("phase") != "B2_calibration_only_no_training":
        raise RuntimeError("only deterministic Phase B2 calibration is authorized")
    for field in ("source_phase_a2_results", "failed_phase_b_results", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact checksum mismatch: {field}")
    a2 = json.loads(resolve(contract["source_phase_a2_results"]).read_text(encoding="utf-8"))
    failed = json.loads(resolve(contract["failed_phase_b_results"]).read_text(encoding="utf-8"))
    if a2["gate_a"]["gate_a_pass"] is not True or failed["gate_b"]["pass"] is not False:
        raise RuntimeError("B2 requires passed Gate A and preserved failed Gate B evidence")
    calibration, gate, opened = map(set, (
        contract["calibration_seeds"], contract["reserved_gate_seeds"], contract["opened_or_prior_seeds"]
    ))
    if calibration - opened or calibration & gate or gate & opened:
        raise RuntimeError("Phase B2 seed cohorts overlap")
    if len(calibration) != 10 or len(gate) != 20:
        raise RuntimeError("Phase B2 requires 10 calibration and 20 gate seeds")
    if int(contract["workers"]) * int(contract["threads_per_worker"]) > 8:
        raise RuntimeError("worker contract exceeds stable host capacity")
    return contract


def expand_candidates(contract: dict) -> list[dict]:
    grid, fixed = contract["candidate_grid"], contract["candidate_grid"]["fixed"]
    rows = []
    for index, (energy, depletion, fairness, stale) in enumerate(itertools.product(
        grid["energy_weight"], grid["depletion_weight"], grid["fairness_guard"], grid["stale_guard"]
    ), start=1):
        rows.append({
            "candidate_id": f"gm_{index:02d}_e{energy:g}_d{depletion:g}_fg{fairness:g}_sg{stale:g}",
            "energy_weight": float(energy), "depletion_weight": float(depletion),
            "fairness_guard": float(fairness), "stale_guard": float(stale),
            **{key: float(value) for key, value in fixed.items()},
        })
    if len(rows) != 18:
        raise RuntimeError("frozen B2 grid must contain 18 candidates")
    return rows


def evaluate_candidate(contract_path: str, candidate: dict) -> dict:
    contract = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), 123_000 + int(candidate["candidate_id"].split("_")[1]))
    runtime = dict(contract); runtime["development_seeds"] = list(contract["calibration_seeds"])
    runtime["observation_schema"] = STEP3_CH_CONTEXT_SCHEMA
    config = GuardedTeacherConfig.from_payload(candidate)
    envs = build_transfer_environments(runtime, contract["scenario"], trace=None)
    rows = []
    for env in envs:
        observation, mask, _ = env.reset(); done = False; energy = 0.0; allocated = 0; violations = 0
        stale_swaps = 0; fairness_swaps = 0; guard_failures = 0; decisions = 0
        while not done:
            _, active, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
            action, audit = guarded_marginal_action(
                env, active, caps, int(contract["budget"]), config, return_audit=True
            )
            decisions += int(audit["target_slots"])
            stale_swaps += len(audit["stale_swaps"])
            fairness_swaps += len(audit["fairness_swaps"])
            guard_failures += int(not audit["stale_guard_satisfied"] or not audit["fairness_guard_satisfied"])
            violations += int(
                int(action.sum()) > int(contract["budget"]) or np.any(action < 0)
                or np.any(action > caps) or np.any(action[~active] != 0)
            )
            allocated += int(action.sum()); observation, mask, done, info = env.step(action)
            role = info["energy_trace"]["role_energy"]
            reconstructed = sum(float(np.asarray(role[name]).sum()) for name in ("member_tx", "ch_rx", "ch_aggregate", "ch_tx_bs", "idle"))
            consumed = float(np.asarray(info["energy_trace"]["consumed"]).sum())
            violations += int(abs(reconstructed - consumed) > 1e-12); energy += consumed
        counts = env.step3_qos_counts; demand = max(1, int(counts["demand"]))
        rows.append({
            "seed": int(env.seed), "target_rank": int(env.target_rank),
            "delivery_ratio": int(counts["delivered"]) / demand,
            "packets_per_j": int(env.base.total_packets) / max(energy, 1e-12),
            "stale_ratio": int(counts["stale"]) / demand,
            "fairness": float(counts["episode_service_fairness"]),
            "rmst": int(env.base.t_fnd if env.base.t_fnd is not None else contract["horizon"]),
            "allocated_slots": allocated, "feasibility_violations": violations,
            "stale_guard_swaps": stale_swaps, "fairness_guard_swaps": fairness_swaps,
            "guard_infeasible_rounds": guard_failures, "decisions": decisions,
        })
    summaries = {}
    fields = ("delivery_ratio", "packets_per_j", "stale_ratio", "fairness", "rmst", "allocated_slots", "stale_guard_swaps", "fairness_guard_swaps", "guard_infeasible_rounds", "decisions")
    for seed in contract["calibration_seeds"]:
        selected = [row for row in rows if row["seed"] == int(seed)]
        summaries[str(seed)] = {key: float(np.mean([row[key] for row in selected])) for key in fields}
        summaries[str(seed)]["feasibility_violations"] = int(sum(row["feasibility_violations"] for row in selected))
    aggregate = {f"mean_{key}": float(np.mean([row[key] for row in summaries.values()])) for key in fields}
    aggregate["feasibility_violations"] = int(sum(row["feasibility_violations"] for row in summaries.values()))
    return {"candidate": candidate, "aggregate": aggregate, "seed_summaries": summaries}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    contract_path, output_dir = resolve(args.contract), resolve(args.output_dir)
    contract = load_contract(contract_path); candidates = expand_candidates(contract)
    output_dir.mkdir(parents=True, exist_ok=True); started = time.perf_counter(); results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
        futures = [pool.submit(evaluate_candidate, str(contract_path), candidate) for candidate in candidates]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result()); print(f"PHASE_B2_CALIBRATION_PROGRESS={index}/{len(futures)}", flush=True)
    selected, passed = select_candidate(results, contract["gate_b_targets"])
    results.sort(key=lambda row: row["candidate"]["candidate_id"])
    selected_payload = {
        "schema_version": 1,
        "status": (
            "guarded_teacher_frozen_after_calibration" if passed
            else "diagnostic_best_candidate_not_gate_eligible"
        ),
        "candidate": selected["candidate"], "calibration_aggregate": selected["aggregate"],
        "calibration_checks": selected["checks"], "calibration_passed_all_targets": passed,
        "normalized_total_violation": selected["normalized_total_violation"],
        "selection_rule": contract["selection_rule"], "calibration_seeds": contract["calibration_seeds"],
        "reserved_gate_seeds_opened": False,
        "gate_b2_authorized": bool(passed),
        "teacher_implementation": "agents/guarded_marginal_teacher.py",
        "teacher_implementation_sha256": sha256("agents/guarded_marginal_teacher.py"), "no_training": True,
    }
    selected_path = output_dir / (
        "selected_teacher.json" if passed else "diagnostic_best_candidate.json"
    )
    selected_path.write_text(json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "status": (
            "phase_b2_calibration_passed" if passed
            else "phase_b2_calibration_failed_stop"
        ),
        "contract_sha256": sha256(contract_path), "runner_sha256": sha256(Path(__file__)),
        "calibration_seeds": contract["calibration_seeds"], "reserved_gate_seeds_opened": False,
        "candidate_count": len(results), "candidates": results,
        "selected_candidate_id": selected["candidate"]["candidate_id"],
        "selected_teacher": str(selected_path), "selected_teacher_sha256": sha256(selected_path),
        "selected_passed_all_calibration_targets": passed,
        "gate_b2_authorized": bool(passed),
        "elapsed_seconds": time.perf_counter() - started, "neural_training_performed": False,
    }
    (output_dir / "calibration_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected": payload["selected_candidate_id"], "calibration_passed": passed, "aggregate": selected["aggregate"], "elapsed_seconds": payload["elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
