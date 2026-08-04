"""Run one preregistered Phase 2B budget-12 confirmation and audit it."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (2299, 3299, 4299)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True, choices=SEEDS)
    parser.add_argument("--episodes", type=int, default=125)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--skip-existing-pass", action="store_true")
    return parser.parse_args()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def original_metrics(checkpoint):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    evaluation = metadata.get("greedy_evaluation")
    if not evaluation:
        raise RuntimeError("initial checkpoint lacks greedy evaluation metadata")
    return evaluation


def acceptance(summary, audit, baseline):
    repaired = audit["repaired"]
    original = audit["original"]
    evaluation = summary["greedy_evaluation"]
    checks = {
        "completed_requested_episodes": (
            summary["episodes_completed"] == summary["episodes_requested"]
        ),
        "full_curriculum_seen": bool(summary["full_curriculum_seen"]),
        "no_nonfinite": not bool(summary["nonfinite_detected"]),
        "no_always_sleep_collapse": not bool(summary["always_sleep_collapse"]),
        "convergence_pass": bool(summary["convergence"]["pass"]),
        "policy_stability_pass": bool(summary["policy_stability"]["pass"]),
        "trajectory_directional": (
            repaired["high_more_slots"] > repaired["high_fewer_slots"]
        ),
        "marginal_ordering_improved": (
            repaired["all_marginals_ordered_fraction"]
            > original["all_marginals_ordered_fraction"]
        ),
        "projected_node_sensitivity_improved": (
            repaired["projected_node_changes_fraction"]
            > original["projected_node_changes_fraction"]
        ),
        "fnd_retention_at_least_95_percent": (
            evaluation["mean_fnd_free_steps"]
            >= 0.95 * baseline["mean_fnd_free_steps"]
        ),
        "delivery_ratio_not_down_more_than_0_02": (
            evaluation["mean_delivery_ratio"]
            >= baseline["mean_delivery_ratio"] - 0.02
        ),
        "queue_fairness_not_decreased": (
            evaluation["mean_queue_fairness"]
            >= baseline["mean_queue_fairness"]
        ),
    }
    return checks, all(checks.values())


def main():
    args = parse_args()
    checkpoint = args.initial_checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    run_name = args.run_name or (
        f"phase2b_confirm_shared_b12_seed{args.seed}_{args.episodes}ep"
    )
    run_dir = ROOT / "outputs" / "phase2" / run_name
    summary_path = run_dir / "summary.json"
    decision_path = run_dir / "confirmation_decision.json"
    if args.skip_existing_pass and decision_path.is_file():
        existing = load_json(decision_path)
        if existing.get("confirmation_pass"):
            print(f"SKIP_CONFIRMATION_PASS run={run_name}")
            return 0

    command = [
        sys.executable,
        "-B",
        "experiments/train_phase2_dynamic_curriculum.py",
        "--episodes", str(args.episodes),
        "--max-steps", "300",
        "--development-seeds", "2300,2301,2302,2303,2304",
        "--optimizer-seed", str(args.seed),
        "--run-name", run_name,
        "--device", args.device,
        "--precision", args.precision,
        "--initial-checkpoint", str(checkpoint),
        "--projection-budget", "12",
        "--architecture", "shared_branching",
        "--learning-rate", "0.00001",
        "--epsilon-start", "0.10",
        "--epsilon-end", "0.03",
        "--normalize-input-blocks",
        "--trajectory-loss-weight", "1.0",
        "--concavity-loss-weight", "0.10",
        "--trajectory-margin-fraction", "0.05",
        "--stability-interval", "25",
        "--stability-tail-episodes", "75",
    ]
    print("TRAIN_COMMAND=" + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    audit_path = run_dir / "mid_episode_hybrid_sensitivity.json"
    subprocess.run(
        [
            sys.executable,
            "-B",
            "experiments/audit_phase2_mid_episode_hybrid_sensitivity.py",
            str(checkpoint),
            str(run_dir / "branching_c51.pt"),
            "--output", str(audit_path),
            "--audit-seed", "20260803",
        ],
        cwd=ROOT,
        check=True,
    )
    summary = load_json(summary_path)
    audit = load_json(audit_path)
    baseline = original_metrics(checkpoint)
    checks, passed = acceptance(summary, audit, baseline)
    decision = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "confirmation_pass" if passed else "confirmation_fail",
        "confirmation_pass": passed,
        "seed": args.seed,
        "episodes": args.episodes,
        "precision": args.precision,
        "run_name": run_name,
        "initial_checkpoint": str(checkpoint),
        "predeclared_checks": checks,
        "baseline_greedy_evaluation": baseline,
        "repaired_greedy_evaluation": summary["greedy_evaluation"],
        "hybrid_sensitivity": {
            "original": audit["original"],
            "repaired": audit["repaired"],
        },
        "scope": (
            "Development confirmation only. This does not replace held-out "
            "evaluation or paired 30-trial statistical reporting."
        ),
    }
    decision_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    print(f"PHASE2B_CONFIRMATION_PASS={passed}")
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())