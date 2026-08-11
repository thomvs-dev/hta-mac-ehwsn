"""Apply frozen five-seed global-QoS and paired-lifetime development gates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def hta_rows(summary_path):
    raw = summary_path.parent / "raw_trials.csv"
    with raw.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["policy"] == "hta_mac"]
    return {int(row["seed"]): row for row in rows}


def lifetime_value(row):
    return float(row["t_fnd"] if row["t_fnd"] not in ("", "None") else row["censor_round"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--reference-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    candidate_summary = json.loads(args.candidate_summary.read_text())
    reference_summary = json.loads(args.reference_summary.read_text())
    candidate, reference = hta_rows(args.candidate_summary), hta_rows(args.reference_summary)
    common = sorted(set(candidate) & set(reference))
    if common != [2400, 2401, 2402, 2403, 2404]:
        raise RuntimeError(f"expected exactly five frozen development pairs, got {common}")
    qos = {}
    deltas = {}
    for seed in common:
        row = candidate[seed]
        qos[seed] = {
            "delivery_ratio": float(row["delivery_ratio"]),
            "stale_drop_ratio": float(row["stale_drop_ratio"]),
            "queue_fairness": float(row["queue_fairness"]),
            "pass": bool(
                float(row["delivery_ratio"]) >= 0.55
                and float(row["stale_drop_ratio"]) <= 0.45
                and float(row["queue_fairness"]) >= 0.70
            ),
        }
        deltas[seed] = lifetime_value(row) - lifetime_value(reference[seed])
    median_delta = float(np.median(list(deltas.values())))
    not_worse = sum(value >= 0.0 for value in deltas.values())
    gates = {
        "candidate_structural": candidate_summary.get("phase3_structural_gate_pass") is True,
        "reference_structural": reference_summary.get("phase3_structural_gate_pass") is True,
        "all_five_global_qos": all(row["pass"] for row in qos.values()),
        "nonnegative_median_paired_fnd": median_delta >= 0.0,
    }
    eligible = all(gates.values())
    payload = {
        "schema_version": 1,
        "status": "development_candidate_eligible" if eligible else "no_candidate_global_qos_feasible",
        "confirmation_seeds_opened": False,
        "thresholds": {"delivery_min": 0.55, "stale_max": 0.45, "fairness_min": 0.70},
        "gates": gates,
        "per_seed_global_qos": qos,
        "paired_fnd_or_censor_delta_rounds": deltas,
        "median_paired_fnd_or_censor_delta_rounds": median_delta,
        "not_worse_fnd_seed_count": not_worse,
        "preferred_not_worse_in_four_of_five": not_worse >= 4,
        "inferential_superiority_claimed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates, "median_fnd_delta": median_delta}, indent=2))
    raise SystemExit(0 if eligible else 3)


if __name__ == "__main__":
    main()
