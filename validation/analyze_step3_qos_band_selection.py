"""Audit the frozen adaptive selection without upgrading it to confirmation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    config = json.loads(config_path.read_text())
    report_path = ROOT / config["selection_report"]
    checkpoint_path = ROOT / config["source_checkpoint"]
    report = json.loads(report_path.read_text())
    selected = next(
        row for row in report["candidates"]
        if float(row["upper_target"]) == float(config["upper_delivery_target"])
    )
    gates = {
        "selection_report_hash_matches": digest(report_path) == config["selection_report_sha256"],
        "checkpoint_hash_matches": digest(checkpoint_path) == config["source_checkpoint_sha256"],
        "report_selected_same_candidate": float(report["selected_upper_target"]) == float(config["upper_delivery_target"]),
        "joint_qos_pass": int(selected["joint_qos_pass_count"]) >= 18,
        "delivery_pass": int(selected["delivery_pass_count"]) == 20,
        "fnd_gate_pass": float(selected["mean_fnd_free_steps"]) >= 1158.75,
        "two_sided_projection_active": int(selected["totals"]["added_slots"]) > 0 and int(selected["totals"]["removed_slots"]) > 0,
        "risk_gate_active": int(selected["totals"]["risk_blocked_slots"]) > 0,
        "confirmation_not_used": report.get("confirmation_seed_2401_used") is False and config.get("confirmation_seed_used") is False,
        "publication_not_claimed": config.get("publication_evidence") is False,
    }
    payload = {
        "schema_version": 1,
        "status": "adaptive_candidate_pass_confirmation_required" if all(gates.values()) else "audit_fail",
        "overall_pass": all(gates.values()),
        "gates": gates,
        "selected_metrics": {
            "joint_qos_pass_count": selected["joint_qos_pass_count"],
            "mean_delivery_ratio": selected["mean_delivery_ratio"],
            "mean_fairness": selected["mean_fairness"],
            "mean_fnd_free_steps": selected["mean_fnd_free_steps"],
            "fnd_gate_margin_rounds": float(selected["mean_fnd_free_steps"]) - 1158.75,
        },
        "method_classification": "shielded_hybrid_rl_with_demonstration_distillation",
        "confirmation_authorized": all(gates.values()),
        "publication_claim_authorized": False,
        "claim_boundary": config["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if payload["overall_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
