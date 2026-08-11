"""Decide whether a 100-episode Step 3 probe warrants any full training."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def lifetime(row):
    return float(row["t_fnd"] if row["t_fnd"] not in ("", "None") else row["censor_round"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase3-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.phase3_summary.read_text())
    with (args.phase3_summary.parent / "raw_trials.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    hta = {int(row["seed"]): row for row in rows if row["policy"] == "hta_mac"}
    energy = {int(row["seed"]): row for row in rows if row["policy"] == "energy_proportional"}
    seeds = sorted(set(hta) & set(energy))
    if seeds != [2400, 2401, 2402, 2403, 2404]:
        raise RuntimeError(f"expected five paired development seeds, got {seeds}")
    qos = [float(hta[s]["delivery_ratio"]) >= 0.55 and float(hta[s]["stale_drop_ratio"]) <= 0.45 and float(hta[s]["queue_fairness"]) >= 0.70 for s in seeds]
    fnd = [lifetime(hta[s]) - lifetime(energy[s]) for s in seeds]
    efficiency = [float(hta[s]["energy_efficiency_packets_per_j"]) / max(float(energy[s]["energy_efficiency_packets_per_j"]), 1e-12) - 1.0 for s in seeds]
    horizon = int(summary["horizon"])
    median_fnd, median_efficiency = float(np.median(fnd)), float(np.median(efficiency))
    gates = {
        "phase3_structural": summary.get("phase3_structural_gate_pass") is True,
        "all_five_global_qos": all(qos),
        "median_fnd_nonnegative": median_fnd >= 0.0,
        "fnd_not_worse_in_four_of_five": sum(value >= 0.0 for value in fnd) >= 4,
        "measurable_gain": median_fnd >= 0.01 * horizon or median_efficiency >= 0.01,
    }
    passed = all(gates.values())
    payload = {
        "schema_version": 1,
        "status": "full_training_may_be_planned" if passed else "bounded_probe_failed_stop_full_training",
        "overall_pass": passed,
        "full_training_started": False,
        "confirmation_seeds_opened": False,
        "gates": gates,
        "paired_fnd_or_censor_delta_rounds": fnd,
        "median_fnd_or_censor_delta_rounds": median_fnd,
        "paired_packets_per_joule_relative_delta": efficiency,
        "median_packets_per_joule_relative_delta": median_efficiency,
        "inferential_claim_allowed": False,
        "next_step_if_pass": "freeze_full_training_contract_in_a_separate_notebook",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "gates": gates}, indent=2))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
