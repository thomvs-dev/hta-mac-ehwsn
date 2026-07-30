"""Audit paired Phase 3 pilot deltas and censoring before Phase 4."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs" / "phase3" / "paired_pilot_5seed"


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def main():
    with (RUN / "raw_trials.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keyed = {(int(row["seed"]), row["policy"]): row for row in rows}
    seeds = sorted({int(row["seed"]) for row in rows})
    comparisons = {}
    for baseline in (
        "static_equal",
        "s2a2mac_adapted",
        "random_budgeted_diagnostic",
    ):
        deltas = {"t_fnd": [], "throughput": [], "queue_fairness": [], "energy_efficiency_packets_per_j": []}
        for seed in seeds:
            learned = keyed[(seed, "hta_mac")]
            other = keyed[(seed, baseline)]
            for metric in deltas:
                deltas[metric].append(
                    float(learned[metric]) - float(other[metric])
                )
        comparisons[baseline] = {
            metric: {
                "paired_deltas": values,
                "median_delta": median(values),
                "hta_wins": int(sum(value > 0.0 for value in values)),
                "ties": int(sum(value == 0.0 for value in values)),
                "hta_losses": int(sum(value < 0.0 for value in values)),
            }
            for metric, values in deltas.items()
        }

    learned_rows = [keyed[(seed, "hta_mac")] for seed in seeds]
    schedule = json.loads((RUN / "summary.json").read_text(encoding="utf-8"))[
        "schedule_metadata"
    ]
    report = {
        "status": "phase3_implemented_phase4_blocked",
        "seeds": seeds,
        "comparisons": comparisons,
        "hta_hnd_observed": int(sum(row["t_hnd"] != "" for row in learned_rows)),
        "hta_hnd_right_censored": int(sum(row["t_hnd"] == "" for row in learned_rows)),
        "all_primary_trials_right_censored": all(
            row["right_censored"] == "True" for row in rows
        ),
        "schedule_coverage_rounds": {
            seed: schedule[str(seed)]["coverage_rounds"] for seed in seeds
        },
        "schedule_stop_reasons": {
            seed: schedule[str(seed)]["stop_reason"] for seed in seeds
        },
        "phase4_blockers": [
            "HTA-MAC loses T_FND and throughput to the S2A2MAC adaptation on all five pilot seeds.",
            "Four of five HTA-MAC HND observations are right-censored, so paired Wilcoxon testing on raw HND is invalid without a censor-aware design.",
            "Every trial is right-censored when the exogenous HEART-CH schedule selects no CH before 3000 rounds.",
            "The previous S2A2MAC novelty sentence is contradicted by the primary source and must not be used.",
        ],
        "phase2_return_recommended": True,
        "inferential_claim_allowed": False,
    }
    output = ROOT / "outputs" / "logs" / "phase3_pilot_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        "HTA_MINUS_STATIC_MEDIAN_FND="
        f"{comparisons['static_equal']['t_fnd']['median_delta']:.4f}"
    )
    print(
        "HTA_MINUS_STATIC_MEDIAN_THROUGHPUT="
        f"{comparisons['static_equal']['throughput']['median_delta']:.4f}"
    )
    print(
        "HTA_MINUS_S2A2_MEDIAN_FND="
        f"{comparisons['s2a2mac_adapted']['t_fnd']['median_delta']:.4f}"
    )
    print(
        "HTA_MINUS_S2A2_MEDIAN_THROUGHPUT="
        f"{comparisons['s2a2mac_adapted']['throughput']['median_delta']:.4f}"
    )
    print(f"HTA_HND_CENSORED={report['hta_hnd_right_censored']}/5")
    print("PHASE4_READY=False")
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
