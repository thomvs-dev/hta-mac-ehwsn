"""Evaluate the frozen deterministic marginal teacher at Gate B."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import subprocess
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


def load_contract(path: Path) -> tuple[dict, dict]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("status") != "frozen_before_v2_phase_b_teacher_gate":
        raise RuntimeError("Gate B contract is not frozen")
    if contract.get("phase") != "B_gate_only_no_training":
        raise RuntimeError("only Gate B evaluation is authorized")
    for field in ("source_calibration_results", "selected_teacher", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact checksum mismatch: {field}")
    selected = json.loads(resolve(contract["selected_teacher"]).read_text(encoding="utf-8"))
    if selected.get("status") != "teacher_frozen_after_calibration":
        raise RuntimeError("selected teacher is not frozen")
    if sha256(selected["teacher_implementation"]) != selected["teacher_implementation_sha256"]:
        raise RuntimeError("teacher implementation changed after calibration")
    gate = set(contract["gate_seeds"])
    prohibited = (
        set(contract["calibration_seeds"]) | set(contract["all_prior_v2_seeds"])
        | set(contract["permanently_opened_confirmation_seeds"])
    )
    if gate & prohibited or len(gate) != 20:
        raise RuntimeError("Gate B seed cohort is not fresh and distinct")
    if int(contract["workers"]) * int(contract["threads_per_worker"]) > 8:
        raise RuntimeError("worker contract exceeds stable host capacity")
    return contract, selected


def evaluate_seed(contract_path: str, seed: int) -> dict:
    contract, selected = load_contract(resolve(contract_path))
    set_cpu_contract(int(contract["threads_per_worker"]), int(seed) + 122_000)
    runtime = dict(contract); runtime["development_seeds"] = [int(seed)]
    runtime["observation_schema"] = STEP3_CH_CONTEXT_SCHEMA
    config = MarginalTeacherConfig.from_payload(selected["candidate"])
    envs = build_transfer_environments(runtime, contract["scenario"], trace=None)
    rows = []
    for env in envs:
        observation, mask, _ = env.reset()
        done = False; energy = 0.0; allocated = 0; violations = 0
        role_totals = {name: 0.0 for name in ("member_tx", "ch_rx", "ch_aggregate", "ch_tx_bs", "idle")}
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
            reconstructed = 0.0
            for name in role_totals:
                value = float(np.asarray(role[name]).sum())
                role_totals[name] += value; reconstructed += value
            consumed = float(np.asarray(info["energy_trace"]["consumed"]).sum())
            violations += int(abs(reconstructed - consumed) > 1e-12)
            energy += consumed
        counts = env.step3_qos_counts; demand = max(1, int(counts["demand"]))
        rows.append({
            "seed": int(seed), "target_rank": int(env.target_rank),
            "delivery_ratio": int(counts["delivered"]) / demand,
            "packets_per_j": int(env.base.total_packets) / max(energy, 1e-12),
            "stale_ratio": int(counts["stale"]) / demand,
            "fairness": float(counts["episode_service_fairness"]),
            "rmst": int(env.base.t_fnd if env.base.t_fnd is not None else contract["horizon"]),
            "global_packets": int(env.base.total_packets), "network_energy_j": energy,
            "allocated_slots": allocated, "feasibility_violations": violations,
            "role_energy_j": role_totals,
        })
    fields = ("delivery_ratio", "packets_per_j", "stale_ratio", "fairness", "rmst", "global_packets", "network_energy_j", "allocated_slots")
    summary = {key: float(np.mean([row[key] for row in rows])) for key in fields}
    summary["feasibility_violations"] = int(sum(row["feasibility_violations"] for row in rows))
    summary["rank_units"] = len(rows)
    summary["mean_role_energy_j"] = {
        name: float(np.mean([row["role_energy_j"][name] for row in rows]))
        for name in role_totals
    }
    return {"seed": int(seed), "summary": summary, "rows": rows}


def bootstrap(seed_summaries: dict[str, dict], contract: dict) -> dict:
    rng = np.random.default_rng(int(contract["inference"]["bootstrap_seed"]))
    resamples = int(contract["inference"]["bootstrap_resamples"])
    confidence = float(contract["inference"]["confidence_level"])
    alpha = 1.0 - confidence
    result = {}
    targets = contract["gate_b_targets"]
    specs = {
        "delivery_ratio": (float(targets["delivery_min"]), 1.0),
        "packets_per_j": (float(targets["packets_per_j_min"]), 1.0),
        "stale_ratio": (float(targets["stale_ratio_max"]), -1.0),
        "fairness": (float(targets["fairness_min"]), 1.0),
        "rmst": (float(targets["rmst_min"]), 1.0),
    }
    for metric, (target, sign) in specs.items():
        values = np.asarray([row[metric] for row in seed_summaries.values()], dtype=np.float64)
        draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
        favorable_margin = sign * (values - target)
        sd = float(favorable_margin.std(ddof=1))
        result[metric] = {
            "mean": float(values.mean()), "target": target,
            "mean_favorable_margin": float(favorable_margin.mean()),
            "bootstrap_95_ci_mean": [float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1.0 - alpha / 2))],
            "standardized_margin_dz": float(favorable_margin.mean() / sd) if sd > 0.0 else None,
        }
    return result


def gate_checks(aggregate: dict, targets: dict) -> dict:
    return {
        "delivery": aggregate["mean_delivery_ratio"] >= float(targets["delivery_min"]),
        "packets_per_j": aggregate["mean_packets_per_j"] >= float(targets["packets_per_j_min"]),
        "stale_ratio": aggregate["mean_stale_ratio"] <= float(targets["stale_ratio_max"]),
        "fairness": aggregate["mean_fairness"] >= float(targets["fairness_min"]),
        "rmst": aggregate["mean_rmst"] >= float(targets["rmst_min"]),
        "feasibility": aggregate["feasibility_violations"] <= int(targets["feasibility_violations_max"]),
    }


def render_report(payload: dict) -> str:
    a = payload["aggregate"]; checks = payload["gate_b"]["checks"]
    lines = [
        "# HTA-MAC V2 deterministic teacher Gate B",
        "",
        f"**Status:** `{payload['status']}`  ",
        f"**Gate B:** `{'PASS' if payload['gate_b']['pass'] else 'FAIL'}`  ",
        f"**Teacher:** `{payload['selected_teacher_id']}`  ",
        "**Scope:** CPU-only deterministic teacher evaluation; no neural training.",
        "",
        "| Metric | Mean | Target | Pass | 95% bootstrap CI |",
        "|---|---:|---:|:---:|---:|",
    ]
    targets = payload["targets"]; inference = payload["inference"]
    rows = (
        ("Delivery", "delivery_ratio", targets["delivery_min"], checks["delivery"]),
        ("Packets/J", "packets_per_j", targets["packets_per_j_min"], checks["packets_per_j"]),
        ("Stale ratio", "stale_ratio", targets["stale_ratio_max"], checks["stale_ratio"]),
        ("Fairness", "fairness", targets["fairness_min"], checks["fairness"]),
        ("RMST", "rmst", targets["rmst_min"], checks["rmst"]),
    )
    for label, metric, target, passed in rows:
        ci = inference[metric]["bootstrap_95_ci_mean"]
        lines.append(f"| {label} | {a['mean_'+metric]:.6f} | {target} | {passed} | [{ci[0]:.6f}, {ci[1]:.6f}] |")
    lines.extend([
        f"| Feasibility violations | {a['feasibility_violations']} | 0 | {checks['feasibility']} | -- |",
        "", "## Decision", "",
    ])
    if payload["gate_b"]["pass"]:
        lines.append("Gate B passed. This run still stops before Phase C or any neural training.")
    else:
        lines.append("Gate B failed. The deterministic teacher did not establish that the primary delivery/efficiency target is reachable while retaining all secondary constraints. Neural training is therefore stopped; the saved calibration candidates define the measured structural frontier for this teacher family.")
    lines.extend([
        "", "The selected coefficients were frozen on calibration seeds 12100--12109 before opening Gate B seeds 12200--12219. Opened confirmation seeds 3900--3919 were not used.",
        "", "No teacher retuning, checkpoint selection, imitation, PPO, C51, or Phase C work occurred after Gate B opened.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    contract_path, output_dir, report_path = resolve(args.contract), resolve(args.output_dir), resolve(args.report)
    contract, selected = load_contract(contract_path); output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); completed = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
        futures = [pool.submit(evaluate_seed, str(contract_path), int(seed)) for seed in contract["gate_seeds"]]
        for index, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            completed.append(future.result()); print(f"PHASE_B_GATE_PROGRESS={index}/{len(futures)}", flush=True)
    seed_summaries = {str(row["seed"]): row["summary"] for row in completed}
    fields = ("delivery_ratio", "packets_per_j", "stale_ratio", "fairness", "rmst", "global_packets", "network_energy_j", "allocated_slots")
    aggregate = {f"mean_{key}": float(np.mean([row[key] for row in seed_summaries.values()])) for key in fields}
    aggregate["feasibility_violations"] = int(sum(row["feasibility_violations"] for row in seed_summaries.values()))
    aggregate["seed_units"] = len(seed_summaries)
    checks = gate_checks(aggregate, contract["gate_b_targets"])
    checks["all_20_seed_units_complete"] = len(seed_summaries) == 20
    checks["no_training"] = True
    passed = all(checks.values())
    payload = {
        "schema_version": 1, "status": "phase_b_gate_pass_stop" if passed else "phase_b_gate_fail_stop",
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "contract_sha256": sha256(contract_path), "runner_sha256": sha256(Path(__file__)),
        "selected_teacher_sha256": sha256(contract["selected_teacher"]),
        "selected_teacher_id": selected["candidate"]["candidate_id"],
        "gate_seeds": contract["gate_seeds"], "targets": contract["gate_b_targets"],
        "seed_summaries": seed_summaries, "raw_rows": [row for task in completed for row in task["rows"]],
        "aggregate": aggregate, "inference": bootstrap(seed_summaries, contract),
        "gate_b": {"checks": checks, "pass": passed, "continuation_authorized": passed},
        "elapsed_seconds": time.perf_counter() - started,
        "selected_teacher_changed_after_gate_open": False,
        "neural_training_performed": False, "phase_c_started": False,
        "claim_boundary": "fresh 20-seed Gate B development evidence in the simulated reference transfer environment",
    }
    results_path = output_dir / "results.json"; results_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "seed_level_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        rows = [{"seed": int(seed), **{key: value for key, value in row.items() if not isinstance(value, dict)}} for seed, row in seed_summaries.items()]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report_path.parent.mkdir(parents=True, exist_ok=True); report_path.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps({"status": payload["status"], "aggregate": aggregate, "checks": checks, "results": str(results_path), "report": str(report_path), "elapsed_seconds": payload["elapsed_seconds"]}, indent=2), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
