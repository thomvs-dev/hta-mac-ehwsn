"""Apply frozen successive-halving gates; never select an infeasible candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    v3 = payload.get("step3_v3", {})
    greedy = payload.get("greedy_evaluation", {}).get("step3_target_qos", {})
    eligible = bool(
        not payload.get("nonfinite_detected", False)
        and not payload.get("always_sleep_collapse", False)
        and payload.get("trajectory_q_check", {}).get("differentiated", False)
        and greedy.get("pass", False)
        and v3.get("risk_non_dominating_pass", False)
        and all(float(value) > 0 for value in v3.get("risk_active_by_seed", {}).values())
    )
    evaluation = payload.get("greedy_evaluation", {})
    return {
        "path": str(path),
        "candidate_id": payload.get("step3_v3", {}).get("candidate_id", path.parent.name),
        "eligible": eligible,
        "qos_joint_pass_fraction": float(greedy.get("joint_pass_fraction", 0.0)),
        "mean_fnd_free_steps": float(evaluation.get("mean_fnd_free_steps", 0.0)),
        "mean_throughput": float(evaluation.get("mean_throughput", 0.0)),
        "mean_packets_per_joule": float(evaluation.get("mean_packets_per_joule", 0.0)),
        "risk_fraction": float(v3.get("risk_tail_absolute_reward_fraction", 0.0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--maximum-survivors", type=int, required=True)
    parser.add_argument("--stage", choices=("25", "100", "250"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [load(path.resolve()) for path in args.summaries]
    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            row["qos_joint_pass_fraction"], row["mean_fnd_free_steps"],
            row["mean_throughput"], row["mean_packets_per_joule"],
        ), reverse=True,
    )
    survivors = eligible[: args.maximum_survivors]
    status = "candidate_selected" if survivors else "no_candidate_global_qos_feasible"
    payload = {
        "schema_version": 1,
        "status": status,
        "stage_episodes": int(args.stage),
        "selection_order": ["qos_feasible", "fnd", "throughput", "packets_per_joule"],
        "least_bad_infeasible_selection_forbidden": True,
        "candidates": rows,
        "survivors": survivors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "survivors": [r["candidate_id"] for r in survivors]}, indent=2))
    raise SystemExit(0 if survivors else 3)


if __name__ == "__main__":
    main()
