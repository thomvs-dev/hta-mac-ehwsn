"""Frozen independent confirmation for the selected listwise residual ranker."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.ch_depletion_risk import validate_ch_risk_config
from agents.qos_constraints_v3 import Step3QoSConstraintConfig
from experiments.diagnose_step3_delivery_feasibility import build_environments, load_agent
from experiments.distill_step3_qos_shield import set_cpu_contract
from experiments.sweep_step3_primary_listwise_residual import (
    SetRemovalRanker,
    evaluate_ranker,
    evaluate_teacher,
)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def load_contract(path):
    contract = json.loads(path.read_text())
    if contract.get("status") != "frozen_before_listwise_residual_confirmation":
        raise RuntimeError("confirmation contract is not frozen")
    for field in ("source_checkpoint", "selected_ranker_checkpoint", "selection_report", "qos_config", "risk_config"):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {field}")
    if set(contract["confirmation_seeds"]).intersection(contract["development_and_prior_seeds"]):
        raise RuntimeError("confirmation seed overlaps a prior cohort")
    return contract


def load_ranker(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ranker = SetRemovalRanker(int(payload["features"]), int(payload["hidden"]))
    ranker.load_state_dict(payload["state_dict"]); ranker.eval()
    return ranker


def worker(contract_path_string, mode):
    contract = load_contract(resolve(contract_path_string))
    set_cpu_contract(int(contract["threads_per_worker"]), 20260812 + int(mode == "residual"))
    risk = validate_ch_risk_config(json.loads(resolve(contract["risk_config"]).read_text()))
    qos = Step3QoSConstraintConfig.from_payload(json.loads(resolve(contract["qos_config"]).read_text()))
    agent, _ = load_agent(resolve(contract["source_checkpoint"]))
    envs, _, _ = build_environments(None, risk, int(contract["horizon"]), seeds=contract["confirmation_seeds"])
    if mode == "teacher":
        return mode, evaluate_teacher(agent, envs, qos, contract)
    ranker = load_ranker(resolve(contract["selected_ranker_checkpoint"]))
    return mode, evaluate_ranker(agent, ranker, envs, qos, contract)


def paired_statistics(teacher, residual, contract):
    teacher_rows = {(r["seed"], r["target_rank"]): r for r in teacher["rows"]}
    residual_rows = {(r["seed"], r["target_rank"]): r for r in residual["rows"]}
    if teacher_rows.keys() != residual_rows.keys():
        raise RuntimeError("paired confirmation cohorts differ")
    rng = np.random.default_rng(int(contract["statistics"]["random_seed"]))
    resamples = int(contract["statistics"]["paired_bootstrap_resamples"])
    alpha = 1.0 - float(contract["statistics"]["confidence_level"])
    output = {}
    for metric in ("fnd_free_steps", "episode_service_fairness", "global_packets_per_j"):
        differences = np.asarray([residual_rows[k][metric] - teacher_rows[k][metric] for k in sorted(teacher_rows)], dtype=np.float64)
        samples = differences[rng.integers(0, len(differences), size=(resamples, len(differences)))].mean(axis=1)
        if np.allclose(differences, 0.0):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(differences, alternative=contract["statistics"]["wilcoxon_alternative"], zero_method="zsplit").pvalue)
        output[metric] = {"mean_paired_difference_residual_minus_teacher": float(differences.mean()),
                          "bootstrap_confidence_interval": [float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))],
                          "wilcoxon_two_sided_p_value": p_value, "pairs": len(differences)}
    return output


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--contract", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    contract_path, output_path = resolve(args.contract), resolve(args.output); contract = load_contract(contract_path); started = time.perf_counter()
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
        futures = [pool.submit(worker, str(contract_path), mode) for mode in ("teacher", "residual")]
        for future in concurrent.futures.as_completed(futures):
            mode, result = future.result(); results[mode] = result; print(f"CONFIRMATION_COMPLETE={mode}", flush=True)
    teacher, residual = results["teacher"], results["residual"]
    reduction = 1.0 - residual["teacher_action_disagreement_l1"] / max(1, teacher["intervention"]["changed_slots_l1"])
    gates = contract["gates"]
    checks = {"joint_qos": residual["joint_qos_pass_count"] >= int(gates["minimum_joint_qos_pairs"]),
              "teacher_correction_reduction": reduction >= float(gates["minimum_teacher_correction_reduction_fraction"]),
              "fnd": residual["mean_fnd_free_steps"] >= teacher["mean_fnd_free_steps"] - float(gates["maximum_mean_fnd_degradation_rounds"]),
              "fairness": residual["mean_episode_service_fairness"] >= teacher["mean_episode_service_fairness"] - float(gates["maximum_fairness_degradation"]),
              "packets_per_j": residual["mean_global_packets_per_j"] >= teacher["mean_global_packets_per_j"] * (1.0 - float(gates["maximum_packets_per_j_degradation_fraction"]))}
    payload = {"schema_version": 1, "status": "listwise_residual_confirmation_passed" if all(checks.values()) else "listwise_residual_confirmation_failed", "contract_sha256": sha256(contract_path), "confirmation_seeds": contract["confirmation_seeds"], "confirmation_seeds_opened": True, "teacher": teacher, "residual": residual, "teacher_correction_reduction_fraction": reduction, "paired_statistics": paired_statistics(teacher, residual, contract), "checks": checks, "gate_pass": all(checks.values()), "elapsed_seconds": time.perf_counter() - started, "claim_boundary": contract["claim_boundary"]}
    output_path.parent.mkdir(parents=True, exist_ok=True); output_path.write_text(json.dumps(payload, indent=2) + "\n"); print(json.dumps({"status": payload["status"], "checks": checks}, indent=2)); print(f"OUTPUT={output_path}"); return 0 if payload["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
