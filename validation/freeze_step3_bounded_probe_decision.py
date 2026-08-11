"""Freeze whether development headroom evidence justifies one bounded RL probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-ranked-evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.energy_ranked_evidence.read_text())
    candidates = []
    for row in evidence["candidates"]:
        deltas = [float(value) for value in row["paired_fnd_or_censor_delta_rounds"]]
        criteria = {
            "all_five_qos_pass": row["all_five_qos_pass"] is True,
            "median_lifetime_noninferior_within_one_percent": float(row["median_fnd_or_censor_delta_rounds"]) >= -0.01 * evidence["horizon"],
            "at_least_two_seeds_gain_two_percent": sum(value >= 0.02 * evidence["horizon"] for value in deltas) >= 2,
            "no_seed_loses_more_than_one_percent": min(deltas) >= -0.01 * evidence["horizon"],
        }
        candidates.append({**row, "bounded_probe_criteria": criteria, "bounded_probe_justified": all(criteria.values())})
    passing = [row for row in candidates if row["bounded_probe_justified"]]
    payload = {
        "schema_version": 1,
        "status": "one_bounded_100_episode_probe_justified" if passing else "stop_rl_training_no_conditional_headroom",
        "overall_pass": bool(passing),
        "selected_diagnostic_policy": passing[0]["policy"] if passing else None,
        "development_informed_gate": True,
        "confirmation_evidence": False,
        "full_training_authorized": False,
        "energy_ranked_evidence_sha256": sha256(args.energy_ranked_evidence),
        "candidates": candidates,
        "claim_boundary": "authorizes_one_optimizer_seed_for_100_episodes_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "selected": payload["selected_diagnostic_policy"]}, indent=2))
    raise SystemExit(0 if passing else 3)


if __name__ == "__main__":
    main()
