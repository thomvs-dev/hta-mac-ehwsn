"""Calibrate the literature-grounded Phase B3 RSG-energy teacher."""

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
from agents.rsg_energy_teacher import RSGEnergyConfig, RSGEnergyTeacher
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.evaluate_step4_publication_evidence import build_transfer_environments
from experiments.run_v2_phase_b_teacher_calibration import select_candidate


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(value: str | Path) -> str:
    return hashlib.sha256(resolve(value).read_bytes()).hexdigest()


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_v2_phase_b3_literature_rsg_energy_calibration":
        raise RuntimeError("B3 contract is not frozen")
    if contract.get("phase") != "B3_calibration_only_no_training":
        raise RuntimeError("only deterministic B3 calibration is authorized")
    for field in ("failed_phase_b2_evidence", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact checksum mismatch: {field}")
    failed = json.loads(resolve(contract["failed_phase_b2_evidence"]).read_text(encoding="utf-8"))
    if failed.get("status") != "phase_b2_failed_stop":
        raise RuntimeError("B3 requires preserved B2 stop evidence")
    development = set(contract["development_seeds"])
    reserved = set(contract["reserved_gate_seeds"])
    opened = set(contract["opened_or_forbidden_seeds"])
    if development - opened or development & reserved or reserved & opened:
        raise RuntimeError("B3 seed protocol is invalid")
    if set(range(3900, 3920)) & development:
        raise RuntimeError("opened manuscript confirmation seeds are forbidden")
    if len(development) != 10 or len(reserved) != 20:
        raise RuntimeError("B3 requires 10 development and 20 reserved seeds")
    if int(contract["workers"]) * int(contract["threads_per_worker"]) > 8:
        raise RuntimeError("worker contract exceeds stable host capacity")
    return contract


def expand_candidates(contract: dict) -> list[dict]:
    grid, fixed = contract["candidate_grid"], contract["candidate_grid"]["fixed"]
    rows = []
    values = itertools.product(
        grid["energy_weight"], grid["deadline_weight"],
        grid["tsls_weight"], grid["harvest_credit"],
    )
    for index, (energy, deadline, tsls, harvest) in enumerate(values, start=1):
        rows.append({
            "candidate_id": f"rsg_{index:02d}_e{energy:g}_dl{deadline:g}_t{tsls:g}_h{harvest:g}",
            "energy_weight": float(energy), "deadline_weight": float(deadline),
            "tsls_weight": float(tsls), "harvest_credit": float(harvest),
            **{key: float(value) for key, value in fixed.items()},
        })
    if len(rows) != 24:
        raise RuntimeError("frozen B3 grid must contain 24 candidates")
    return rows


def evaluate_candidate(contract_path: str, candidate: dict) -> dict:
    contract = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), 125_000 + int(candidate["candidate_id"].split("_")[1]))
    runtime = dict(contract)
    runtime["development_seeds"] = list(contract["development_seeds"])
    runtime["observation_schema"] = STEP3_CH_CONTEXT_SCHEMA
    config = RSGEnergyConfig.from_payload(candidate)
    envs = build_transfer_environments(runtime, contract["scenario"], trace=None)
    rows = []
    for env in envs:
        controller = RSGEnergyTeacher(env.base.n_nodes, config)
        observation, mask, _ = env.reset()
        done = False; energy = 0.0; allocated = 0; violations = 0
        stale_swaps = 0; fairness_swaps = 0; guard_infeasible = 0; max_tsls = 0
        while not done:
            _, active, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
            action, audit = controller.action(
                env, active, caps, int(contract["budget"]), return_audit=True
            )
            stale_swaps += int(audit["stale_swaps"])
            fairness_swaps += int(audit["fairness_swaps"])
            guard_infeasible += int(not audit["stale_guard_satisfied"] or not audit["fairness_guard_satisfied"])
            max_tsls = max(max_tsls, int(audit["max_tsls"]))
            violations += int(
                int(action.sum()) > int(contract["budget"]) or np.any(action < 0)
                or np.any(action > caps) or np.any(action[~active] != 0)
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
            "stale_guard_swaps": stale_swaps, "fairness_guard_swaps": fairness_swaps,
            "guard_infeasible_rounds": guard_infeasible, "max_tsls": max_tsls,
        })
    fields = (
        "delivery_ratio", "packets_per_j", "stale_ratio", "fairness", "rmst",
        "allocated_slots", "stale_guard_swaps", "fairness_guard_swaps",
        "guard_infeasible_rounds", "max_tsls",
    )
    seed_summaries = {}
    for seed in contract["development_seeds"]:
        selected = [row for row in rows if row["seed"] == int(seed)]
        seed_summaries[str(seed)] = {
            key: float(np.mean([row[key] for row in selected])) for key in fields
        } | {"feasibility_violations": int(sum(row["feasibility_violations"] for row in selected))}
    aggregate = {
        f"mean_{key}": float(np.mean([row[key] for row in seed_summaries.values()]))
        for key in fields
    }
    aggregate["feasibility_violations"] = int(sum(row["feasibility_violations"] for row in seed_summaries.values()))
    return {"candidate": candidate, "aggregate": aggregate, "seed_summaries": seed_summaries}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract_path, output_dir = resolve(args.contract), resolve(args.output_dir)
    contract = load_contract(contract_path); candidates = expand_candidates(contract)
    output_dir.mkdir(parents=True, exist_ok=True); started = time.perf_counter(); results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
        futures = [pool.submit(evaluate_candidate, str(contract_path), candidate) for candidate in candidates]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            results.append(future.result())
            print(f"PHASE_B3_CALIBRATION_PROGRESS={index}/{len(futures)}", flush=True)
    selected, passed = select_candidate(results, contract["gate_b_targets"])
    results.sort(key=lambda row: row["candidate"]["candidate_id"])
    selected_payload = {
        "schema_version": 1,
        "status": "rsg_energy_teacher_frozen" if passed else "diagnostic_best_candidate_not_gate_eligible",
        "candidate": selected["candidate"], "development_aggregate": selected["aggregate"],
        "development_checks": selected["checks"], "development_passed_all_targets": passed,
        "normalized_total_violation": selected["normalized_total_violation"],
        "reserved_gate_seeds_opened": False, "gate_b3_authorized": bool(passed),
        "teacher_implementation": "agents/rsg_energy_teacher.py",
        "teacher_implementation_sha256": sha256("agents/rsg_energy_teacher.py"),
        "guard_dependency_sha256": sha256("agents/guarded_marginal_teacher.py"),
        "no_training": True,
    }
    selected_path = output_dir / ("selected_teacher.json" if passed else "diagnostic_best_candidate.json")
    selected_path.write_text(json.dumps(selected_payload, indent=2) + "\n", encoding="utf-8")
    payload = {
        "schema_version": 1,
        "status": "phase_b3_development_passed" if passed else "phase_b3_development_failed_stop",
        "contract_sha256": sha256(contract_path), "runner_sha256": sha256(Path(__file__)),
        "development_seeds": contract["development_seeds"], "reserved_gate_seeds_opened": False,
        "candidate_count": len(results), "candidates": results,
        "selected_candidate_id": selected["candidate"]["candidate_id"],
        "selected_artifact": str(selected_path), "selected_artifact_sha256": sha256(selected_path),
        "selected_passed_all_targets": passed, "gate_b3_authorized": bool(passed),
        "elapsed_seconds": time.perf_counter() - started, "neural_training_performed": False,
    }
    (output_dir / "development_results.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": payload["status"], "selected": payload["selected_candidate_id"],
        "passed": passed, "aggregate": selected["aggregate"],
        "elapsed_seconds": payload["elapsed_seconds"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
