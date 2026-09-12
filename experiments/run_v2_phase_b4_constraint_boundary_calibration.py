"""Calibrate finite-horizon constraint-boundary tracking for Phase B4."""

from __future__ import annotations

import argparse
import concurrent.futures
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
from experiments.run_v2_phase_b_teacher_calibration import select_candidate
from experiments.run_v2_phase_b2_guarded_teacher_calibration import resolve, sha256


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_v2_phase_b4_constraint_boundary_calibration":
        raise RuntimeError("B4 contract is not frozen")
    if contract.get("phase") != "B4_calibration_only_no_training":
        raise RuntimeError("only deterministic B4 calibration is authorized")
    for field in ("failed_b2_evidence", "failed_b3_evidence", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact checksum mismatch: {field}")
    development, reserved, unavailable = map(set, (
        contract["development_seeds"], contract["reserved_gate_seeds"], contract["unavailable_seeds"]
    ))
    if development - unavailable or development & reserved or reserved & unavailable:
        raise RuntimeError("B4 seed protocol is invalid")
    if set(range(3900, 3920)) & development or len(development) != 10 or len(reserved) != 20:
        raise RuntimeError("B4 seed cohorts violate the frozen protocol")
    if int(contract["workers"]) * int(contract["threads_per_worker"]) > 8:
        raise RuntimeError("worker contract exceeds stable host capacity")
    return contract


def expand_candidates(contract: dict) -> list[dict]:
    grid, fixed = contract["candidate_grid"], contract["candidate_grid"]["fixed"]
    rows = []
    for index, (energy, depletion, trigger) in enumerate(itertools.product(
        grid["energy_weight"], grid["depletion_weight"], grid["fairness_control_trigger"]
    ), start=1):
        rows.append({
            "candidate_id": f"cb_{index:02d}_e{energy:g}_d{depletion:g}_fc{trigger:g}",
            "energy_weight": float(energy), "depletion_weight": float(depletion),
            "fairness_guard": float(trigger), "stale_guard": float(fixed["stale_guard"]),
            "reserve_floor": float(fixed["reserve_floor"]),
        })
    if len(rows) != 16:
        raise RuntimeError("frozen B4 grid must contain 16 candidates")
    return rows


def evaluate_candidate(contract_path: str, candidate: dict) -> dict:
    contract = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), 126_000 + int(candidate["candidate_id"].split("_")[1]))
    runtime = dict(contract); runtime["development_seeds"] = list(contract["development_seeds"])
    runtime["observation_schema"] = STEP3_CH_CONTEXT_SCHEMA
    config = GuardedTeacherConfig.from_payload(candidate)
    envs = build_transfer_environments(runtime, contract["scenario"], trace=None)
    rows = []
    for env in envs:
        observation, mask, _ = env.reset(); done = False; energy = 0.0; violations = 0
        allocated = 0; stale_swaps = 0; fairness_swaps = 0; guard_infeasible = 0
        while not done:
            _, active, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
            action, audit = guarded_marginal_action(env, active, caps, int(contract["budget"]), config, return_audit=True)
            stale_swaps += len(audit["stale_swaps"]); fairness_swaps += len(audit["fairness_swaps"])
            guard_infeasible += int(not audit["stale_guard_satisfied"] or not audit["fairness_guard_satisfied"])
            violations += int(int(action.sum()) > int(contract["budget"]) or np.any(action < 0) or np.any(action > caps) or np.any(action[~active] != 0))
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
            "guard_infeasible_rounds": guard_infeasible,
        })
    fields = ("delivery_ratio", "packets_per_j", "stale_ratio", "fairness", "rmst", "allocated_slots", "stale_guard_swaps", "fairness_guard_swaps", "guard_infeasible_rounds")
    summaries = {}
    for seed in contract["development_seeds"]:
        selected = [row for row in rows if row["seed"] == int(seed)]
        summaries[str(seed)] = {key: float(np.mean([row[key] for row in selected])) for key in fields}
        summaries[str(seed)]["feasibility_violations"] = int(sum(row["feasibility_violations"] for row in selected))
    aggregate = {f"mean_{key}": float(np.mean([row[key] for row in summaries.values()])) for key in fields}
    aggregate["feasibility_violations"] = int(sum(row["feasibility_violations"] for row in summaries.values()))
    return {"candidate": candidate, "aggregate": aggregate, "seed_summaries": summaries}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(); contract_path, output_dir = resolve(args.contract), resolve(args.output_dir)
    contract = load_contract(contract_path); candidates = expand_candidates(contract)
    output_dir.mkdir(parents=True, exist_ok=True); started = time.perf_counter(); results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
        futures = [pool.submit(evaluate_candidate, str(contract_path), candidate) for candidate in candidates]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result()); print(f"PHASE_B4_PROGRESS={index}/{len(futures)}", flush=True)
    selected, passed = select_candidate(results, contract["gate_targets_unchanged"]); results.sort(key=lambda row: row["candidate"]["candidate_id"])
    selected_payload = {
        "schema_version": 1, "status": "boundary_teacher_frozen" if passed else "diagnostic_best_candidate_not_gate_eligible",
        "candidate": selected["candidate"], "aggregate": selected["aggregate"], "checks": selected["checks"],
        "passed_all_unchanged_targets": passed, "reserved_gate_seeds_opened": False,
        "gate_b4_authorized": bool(passed), "neural_training_performed": False,
        "teacher_sha256": sha256("agents/guarded_marginal_teacher.py"),
    }
    selected_path = output_dir / ("selected_teacher.json" if passed else "diagnostic_best_candidate.json")
    selected_path.write_text(json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8")
    payload = {
        "schema_version": 1, "status": "phase_b4_development_passed" if passed else "phase_b4_development_failed_stop",
        "contract_sha256": sha256(contract_path), "runner_sha256": sha256(Path(__file__)),
        "candidate_count": len(results), "candidates": results,
        "selected_candidate_id": selected["candidate"]["candidate_id"], "selected_artifact": str(selected_path),
        "selected_artifact_sha256": sha256(selected_path), "selected_passed_all_targets": passed,
        "gate_b4_authorized": bool(passed), "reserved_gate_seeds_opened": False,
        "elapsed_seconds": time.perf_counter() - started, "neural_training_performed": False,
    }
    (output_dir / "development_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selected": payload["selected_candidate_id"], "passed": passed, "aggregate": selected["aggregate"], "elapsed_seconds": payload["elapsed_seconds"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
