"""Controller-aware audit without retroactively changing the legacy Step 3 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--bounded-gate", type=Path, required=True)
    parser.add_argument("--controller-off-diagnostic", type=Path, required=True)
    parser.add_argument("--controller-config", type=Path, required=True)
    parser.add_argument("--cpu-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    bounded = json.loads(args.bounded_gate.read_text())
    off = json.loads(args.controller_off_diagnostic.read_text())
    config = json.loads(args.controller_config.read_text())
    cpu = json.loads(args.cpu_summary.read_text())
    snapshots = summary["policy_stability_snapshots"]
    last_three = snapshots[-3:]
    final = snapshots[-1]["evaluation"]
    on_qos = final["step3_target_qos"]
    off_agg = off["results"]["trained_greedy"]["aggregate"]
    minimum_fnd = float(config["selection_evidence"]["minimum_allowed_mean_fnd_free_steps"])
    controller = summary.get("qos_deficit_override", {})
    gates = {
        "legacy_full_gate_failure_preserved": summary.get("phase2_curriculum_gate_pass") is False,
        "bounded_checkpoint_gate_pass": bounded.get("overall_pass") is True,
        "last_three_controller_on_qos_pass": len(last_three) == 3 and all(
            row["evaluation"]["step3_target_qos"]["pass"] is True for row in last_three
        ),
        "final_controller_on_joint_qos_18_of_20": int(on_qos["joint_pass_count"]) >= 18,
        "final_fnd_noninferior": float(final["mean_fnd_free_steps"]) >= minimum_fnd,
        "controller_applied_during_training_and_evaluation": controller.get("applied_during_training_and_greedy_evaluation") is True,
        "controller_active": int(controller.get("totals", {}).get("added_slots", 0)) > 0,
        "controller_risk_gate_active": int(controller.get("totals", {}).get("risk_blocked_slots", 0)) > 0,
        "controller_off_qos_fails": int(off_agg["joint_qos_pass_count"]) == 0,
        "cpu_evidence_valid": int(cpu["excluded_sample_count"]) == 1 and float(cpu["mean_total_machine_cpu_percent"]) > 70.0,
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "status": "adaptive_shielded_rl_evidence_pass_confirmation_required" if passed else "adaptive_shielded_rl_evidence_fail_stop",
        "overall_pass": passed,
        "legacy_step3_overall_gate_pass": False,
        "legacy_penalty_geometry_failure_reason": "controller_kept_delivery_feasible_so_active_lagrangian_penalty_fraction_was_zero",
        "method_classification": "shielded_hybrid_rl_not_neural_policy_only_constrained_rl",
        "gates": gates,
        "controller_on": {
            "final_joint_qos_passes": int(on_qos["joint_pass_count"]),
            "final_mean_fnd_free_steps": float(final["mean_fnd_free_steps"]),
            "final_mean_throughput": float(final["mean_throughput"]),
            "final_mean_packets_per_joule": float(final["mean_packets_per_joule"]),
        },
        "controller_off": {
            "joint_qos_passes": int(off_agg["joint_qos_pass_count"]),
            "mean_delivery_ratio": float(off_agg["macro_mean_delivery_ratio"]),
            "mean_fnd_free_steps": float(off_agg["mean_fnd_free_steps"]),
        },
        "controller_dependence": {
            "joint_qos_pass_delta": int(on_qos["joint_pass_count"]) - int(off_agg["joint_qos_pass_count"]),
            "claim": "qos_feasibility_is_controller_dependent_in_this_adaptive_development_run",
        },
        "confirmation_run_authorized": passed,
        "held_out_evaluation_authorized": False,
        "publication_claim_authorized": False,
        "evidence_sha256": {
            "summary": digest(args.summary),
            "bounded_gate": digest(args.bounded_gate),
            "controller_off_diagnostic": digest(args.controller_off_diagnostic),
            "controller_config": digest(args.controller_config),
            "cpu_summary": digest(args.cpu_summary),
        },
        "claim_boundary": "adaptive_seed2400_result_requires_fresh_unused_development_seed_confirmation_before_held_out_evaluation",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
