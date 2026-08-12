"""Fresh-cohort ablation and profiling for the confirmed residual ranker."""

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
    base_and_added,
    evaluate_ranker,
    evaluate_teacher,
    teacher_winner,
)
from experiments.sweep_step3_qos_band_projection import qos_band_projection


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path):
    return hashlib.sha256(resolve(path).read_bytes()).hexdigest()


def load_contract(path):
    contract = json.loads(path.read_text())
    if contract.get("status") != "frozen_before_primary_listwise_residual_ablation":
        raise RuntimeError("ablation contract is not frozen")
    for field in (
        "source_checkpoint", "selected_ranker_checkpoint",
        "prior_confirmation_report", "qos_config", "risk_config",
    ):
        if sha256(contract[field]) != contract[f"{field}_sha256"]:
            raise RuntimeError(f"artifact hash mismatch: {field}")
    if set(contract["ablation_seeds"]).intersection(contract["all_prior_seeds"]):
        raise RuntimeError("ablation cohort overlaps prior evidence")
    if int(contract["workers"]) * int(contract["threads_per_worker"]) > 16:
        raise RuntimeError("ablation CPU contract exceeds the frozen allocation")
    return contract


def load_ranker(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    ranker = SetRemovalRanker(int(payload["features"]), int(payload["hidden"]))
    ranker.load_state_dict(payload["state_dict"])
    ranker.eval()
    return ranker, payload


def summarize_rows(rows, intervention):
    return {
        "pairs": len(rows),
        "joint_qos_pass_count": int(sum(row["joint_qos_pass"] for row in rows)),
        "mean_delivery_ratio": float(np.mean([row["delivery_ratio"] for row in rows])),
        "mean_stale_ratio": float(np.mean([row["stale_ratio"] for row in rows])),
        "mean_episode_service_fairness": float(np.mean([row["episode_service_fairness"] for row in rows])),
        "mean_fnd_free_steps": float(np.mean([row["fnd_free_steps"] for row in rows])),
        "mean_global_packets_per_j": float(np.mean([row["global_packets_per_j"] for row in rows])),
        "intervention": intervention,
        "rows": rows,
    }


def evaluate_no_upper(agent, environments, qos, contract):
    rows = []
    audit_totals = {
        "steps": 0, "teacher_action_disagreement_l1": 0,
        "teacher_removed_slots": 0, "executed_slots": 0,
    }
    for env in environments:
        observation, mask, _ = env.reset()
        done = False
        energy = 0.0
        while not done:
            _, padded_mask, caps, base, q_values, added = base_and_added(
                agent, env, observation, mask, contract
            )
            band = contract["band"]
            teacher, teacher_audit = qos_band_projection(
                base, q_values, caps, padded_mask, env,
                lower_target=float(band["lower_delivery_target"]),
                upper_target=float(band["upper_delivery_target"]),
                reserve_floor=float(band["reserve_floor"]),
                completion_fraction=float(band["completion_fraction"]),
            )
            audit_totals["steps"] += 1
            audit_totals["teacher_action_disagreement_l1"] += int(
                np.abs(added - teacher).sum()
            )
            audit_totals["teacher_removed_slots"] += int(teacher_audit["removed"])
            audit_totals["executed_slots"] += int(added.sum())
            observation, mask, done, info = env.step(added)
            energy += float(np.asarray(info["energy_trace"]["consumed"]).sum())
        counts = env.step3_qos_counts
        demand = max(1, int(counts["demand"]))
        delivery = int(counts["delivered"]) / demand
        stale = int(counts["stale"]) / demand
        fairness = float(counts["episode_service_fairness"])
        rows.append({
            "seed": int(env.seed), "target_rank": int(env.target_rank),
            "delivery_ratio": delivery, "stale_ratio": stale,
            "episode_service_fairness": fairness,
            "joint_qos_pass": bool(
                delivery >= qos.minimum_delivery_ratio
                and stale <= qos.maximum_stale_drop_ratio
                and fairness >= qos.minimum_queue_fairness
            ),
            "fnd_free_steps": int(
                env.base.t_fnd if env.base.t_fnd is not None else contract["horizon"]
            ),
            "global_packets": int(env.base.total_packets),
            "network_energy_j": energy,
            "global_packets_per_j": int(env.base.total_packets) / max(energy, 1e-12),
        })
    return summarize_rows(rows, audit_totals)


def worker(contract_path_string, arm):
    contract = load_contract(resolve(contract_path_string))
    set_cpu_contract(
        int(contract["threads_per_worker"]),
        int(contract["statistics"]["random_seed"]) + contract["arms"].index(arm),
    )
    risk = validate_ch_risk_config(json.loads(resolve(contract["risk_config"]).read_text()))
    qos = Step3QoSConstraintConfig.from_payload(json.loads(resolve(contract["qos_config"]).read_text()))
    agent, _ = load_agent(resolve(contract["source_checkpoint"]))
    environments, _, _ = build_environments(
        None, risk, int(contract["horizon"]), seeds=contract["ablation_seeds"]
    )
    if arm == "analytic_teacher":
        return arm, evaluate_teacher(agent, environments, qos, contract)
    if arm == "learned_listwise_residual":
        ranker, _ = load_ranker(resolve(contract["selected_ranker_checkpoint"]))
        return arm, evaluate_ranker(agent, ranker, environments, qos, contract)
    if arm == "no_upper_band_removal":
        return arm, evaluate_no_upper(agent, environments, qos, contract)
    raise ValueError(f"unsupported ablation arm: {arm}")


def paired_statistics(reference, treatment, contract):
    reference_rows = {(r["seed"], r["target_rank"]): r for r in reference["rows"]}
    treatment_rows = {(r["seed"], r["target_rank"]): r for r in treatment["rows"]}
    if reference_rows.keys() != treatment_rows.keys():
        raise RuntimeError("paired ablation cohorts differ")
    rng = np.random.default_rng(int(contract["statistics"]["random_seed"]))
    resamples = int(contract["statistics"]["paired_bootstrap_resamples"])
    alpha = 1.0 - float(contract["statistics"]["confidence_level"])
    output = {}
    metrics = (
        "delivery_ratio", "stale_ratio", "fnd_free_steps",
        "episode_service_fairness", "global_packets_per_j",
    )
    for metric in metrics:
        differences = np.asarray([
            treatment_rows[key][metric] - reference_rows[key][metric]
            for key in sorted(reference_rows)
        ], dtype=np.float64)
        bootstrap = differences[
            rng.integers(0, len(differences), size=(resamples, len(differences)))
        ].mean(axis=1)
        if np.allclose(differences, 0.0):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(
                differences,
                alternative=contract["statistics"]["wilcoxon_alternative"],
                zero_method="zsplit",
            ).pvalue)
        standard_deviation = float(differences.std(ddof=1))
        effect_size = (
            float(differences.mean() / standard_deviation)
            if standard_deviation > 0.0 else None
        )
        output[metric] = {
            "mean_paired_difference_treatment_minus_teacher": float(differences.mean()),
            "median_paired_difference": float(np.median(differences)),
            "bootstrap_confidence_interval": [
                float(np.quantile(bootstrap, alpha / 2)),
                float(np.quantile(bootstrap, 1 - alpha / 2)),
            ],
            "wilcoxon_two_sided_p_value": p_value,
            "paired_cohens_dz": effect_size,
            "wins_ties_losses": {
                "wins": int((differences > 0.0).sum()),
                "ties": int(np.isclose(differences, 0.0).sum()),
                "losses": int((differences < 0.0).sum()),
            },
            "pairs": len(differences),
        }
    return output


def latency_summary(samples_ns):
    samples_us = np.asarray(samples_ns, dtype=np.float64) / 1000.0
    return {
        "median_us": float(np.median(samples_us)),
        "p95_us": float(np.quantile(samples_us, 0.95)),
        "mean_us": float(samples_us.mean()),
        "iterations": len(samples_us),
    }


def profile_components(contract):
    profile = contract["profiling"]
    torch.set_num_threads(int(profile["torch_threads"]))
    ranker, checkpoint = load_ranker(resolve(contract["selected_ranker_checkpoint"]))
    agent, _ = load_agent(resolve(contract["source_checkpoint"]))
    generator = torch.Generator().manual_seed(int(contract["statistics"]["random_seed"]))
    features = torch.randn(
        int(profile["active_nodes"]), int(checkpoint["features"]), generator=generator
    )
    service = np.linspace(0.0, 50.0, int(profile["active_nodes"]), dtype=np.float64)
    action = np.ones(int(profile["active_nodes"]), dtype=np.int64)
    mask = np.ones(int(profile["active_nodes"]), dtype=bool)
    q_values = np.stack((
        np.zeros_like(service), np.linspace(0.1, 1.0, len(service)),
        np.linspace(0.3, 1.3, len(service)), np.linspace(0.5, 1.5, len(service)),
    ), axis=1)
    environment = type("ProfileEnvironment", (), {"cumulative_service": service})()

    def residual_call():
        with torch.no_grad():
            return int(ranker(features).argmax().item())

    def analytic_call():
        return teacher_winner(environment, q_values, action, mask)

    warmup = int(profile["warmup_iterations"])
    iterations = int(profile["timed_iterations"])
    for _ in range(warmup):
        residual_call(); analytic_call()
    residual_times = []
    analytic_times = []
    for _ in range(iterations):
        started = time.perf_counter_ns(); residual_call(); residual_times.append(time.perf_counter_ns() - started)
        started = time.perf_counter_ns(); analytic_call(); analytic_times.append(time.perf_counter_ns() - started)
    ranker_parameters = int(sum(parameter.numel() for parameter in ranker.parameters()))
    base_parameters = int(sum(parameter.numel() for parameter in agent.online.parameters()))
    hidden = int(checkpoint["hidden"])
    features_count = int(checkpoint["features"])
    macs_per_node = features_count if hidden <= 0 else features_count * hidden + hidden
    return {
        "ranker_parameters": ranker_parameters,
        "base_controller_parameters": base_parameters,
        "ranker_parameter_overhead_fraction": ranker_parameters / max(1, base_parameters),
        "ranker_checkpoint_bytes": resolve(contract["selected_ranker_checkpoint"]).stat().st_size,
        "estimated_ranker_macs_per_node": macs_per_node,
        "estimated_ranker_macs_at_profile_node_count": macs_per_node * int(profile["active_nodes"]),
        "component_only_latency_note": "single removal-choice component; excludes environment, C51 inference, QoS-count arithmetic, and repeated multi-removal projection",
        "learned_ranker_latency": latency_summary(residual_times),
        "analytic_teacher_ranking_latency": latency_summary(analytic_times),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract_path, output_path = resolve(args.contract), resolve(args.output)
    contract = load_contract(contract_path)
    started = time.perf_counter()
    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=int(contract["workers"])) as pool:
        futures = [pool.submit(worker, str(contract_path), arm) for arm in contract["arms"]]
        for future in concurrent.futures.as_completed(futures):
            arm, result = future.result()
            results[arm] = result
            print(f"ABLATION_COMPLETE={arm}", flush=True)
    teacher = results["analytic_teacher"]
    residual = results["learned_listwise_residual"]
    no_upper = results["no_upper_band_removal"]
    correction_reduction = 1.0 - residual["teacher_action_disagreement_l1"] / max(
        1, teacher["intervention"]["changed_slots_l1"]
    )
    gates = contract["integrity_gates"]
    checks = {
        "residual_joint_qos": residual["joint_qos_pass_count"] >= int(gates["minimum_residual_joint_qos_pairs"]),
        "residual_teacher_correction_reduction": correction_reduction >= float(gates["minimum_residual_teacher_correction_reduction_fraction"]),
        "residual_fnd": residual["mean_fnd_free_steps"] >= teacher["mean_fnd_free_steps"] - float(gates["maximum_residual_mean_fnd_degradation_rounds"]),
        "residual_fairness": residual["mean_episode_service_fairness"] >= teacher["mean_episode_service_fairness"] - float(gates["maximum_residual_fairness_degradation"]),
        "residual_packets_per_j": residual["mean_global_packets_per_j"] >= teacher["mean_global_packets_per_j"] * (1.0 - float(gates["maximum_residual_packets_per_j_degradation_fraction"])),
        "no_upper_action_distinct": no_upper["intervention"]["teacher_action_disagreement_l1"] >= int(gates["minimum_no_upper_teacher_action_disagreement_l1"]),
    }
    payload = {
        "schema_version": 1,
        "status": "primary_listwise_residual_ablation_passed" if all(checks.values()) else "primary_listwise_residual_ablation_failed",
        "contract_sha256": sha256(contract_path),
        "ablation_seeds": contract["ablation_seeds"],
        "ablation_seeds_opened": True,
        "arms": results,
        "residual_teacher_correction_reduction_fraction": correction_reduction,
        "paired_statistics": {
            "learned_listwise_residual_minus_teacher": paired_statistics(teacher, residual, contract),
            "no_upper_band_removal_minus_teacher": paired_statistics(teacher, no_upper, contract),
        },
        "profiling": profile_components(contract),
        "checks": checks,
        "gate_pass": all(checks.values()),
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": contract["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "checks": checks}, indent=2))
    print(f"OUTPUT={output_path}")
    return 0 if payload["gate_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
