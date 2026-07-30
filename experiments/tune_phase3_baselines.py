"""Tune adapted baseline parameters on dedicated schema-v2 development seeds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines import (
    EnergyProportionalPolicy,
    FFSSAdaptedPolicy,
    HarvestProportionalPolicy,
    S2A2MACAdaptedPolicy,
)
from experiments.run_phase3_pilot import build_assets, run_one, schedule_bundle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="2500,2501,2502,2503,2504")
    parser.add_argument("--horizon", type=int, default=3000)
    parser.add_argument("--run-name", default="baseline_tuning_schema_v2")
    return parser.parse_args()


def candidates():
    rows = []
    for exponent in (0.5, 1.0, 2.0):
        rows.append(
            (
                "energy_proportional",
                {"score_exponent": exponent},
                lambda value=exponent: EnergyProportionalPolicy(value),
            )
        )
        rows.append(
            (
                "harvest_proportional",
                {"score_exponent": exponent},
                lambda value=exponent: HarvestProportionalPolicy(value),
            )
        )
    for weight in (0.25, 0.5, 0.75):
        rows.append(
            (
                "s2a2mac_adapted",
                {"energy_weight": weight, "load_weight": 1.0 - weight},
                lambda value=weight: S2A2MACAdaptedPolicy(value),
            )
        )
    for margin_weight, queue_weight in (
        (1.0, 0.0),
        (0.75, 0.25),
        (0.5, 0.5),
        (0.25, 0.75),
        (0.0, 1.0),
    ):
        rows.append(
            (
                "ffss_adapted",
                {
                    "margin_weight": margin_weight,
                    "queue_weight": queue_weight,
                },
                lambda m=margin_weight, q=queue_weight: FFSSAdaptedPolicy(m, q),
            )
        )
    return rows


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def summarize_configuration(rows):
    event_free = [
        row["t_fnd"] if row["t_fnd_event_observed"] else row["censor_round"]
        for row in rows
    ]
    return {
        "median_fnd_or_censor_round": median(event_free),
        "median_delivery_ratio": median([row["delivery_ratio"] for row in rows]),
        "median_queue_fairness": median([row["queue_fairness"] for row in rows]),
        "median_residual_energy_fairness": median(
            [row["residual_energy_fairness"] for row in rows]
        ),
        "median_energy_efficiency_packets_per_j": median(
            [row["energy_efficiency_packets_per_j"] for row in rows]
        ),
        "fnd_events": int(sum(row["t_fnd_event_observed"] for row in rows)),
    }


def selection_key(policy_name, summary):
    if policy_name == "ffss_adapted":
        return (
            summary["median_delivery_ratio"],
            summary["median_queue_fairness"],
            summary["median_fnd_or_censor_round"],
        )
    return (
        summary["median_fnd_or_censor_round"],
        summary["median_delivery_ratio"],
        summary["median_queue_fairness"],
    )


def main():
    args = parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one tuning seed is required")
    forbidden = set(range(2300, 2305)) | set(range(2400, 2405)) | set(range(3100, 3105))
    overlap = forbidden.intersection(seeds)
    if overlap:
        raise ValueError(f"baseline tuning seeds overlap protected sets: {sorted(overlap)}")

    frozen, solar, thermal, radio, config, manifest = build_assets(args.horizon)
    checkpoint_sha = manifest["checkpoint"]["sha256"]
    bundles = {}
    schedule_metadata = {}
    for seed in seeds:
        bundle, metadata = schedule_bundle(
            frozen, seed, args.horizon, checkpoint_sha
        )
        bundles[seed] = bundle
        schedule_metadata[str(seed)] = metadata

    raw_rows = []
    summaries = []
    for policy_name, parameters, factory in candidates():
        configuration_id = policy_name + ":" + ",".join(
            f"{key}={value}" for key, value in sorted(parameters.items())
        )
        configuration_rows = []
        for seed in seeds:
            row = run_one(
                factory(),
                seed,
                bundles[seed],
                solar,
                thermal,
                radio,
                config,
                True,
            )
            row["configuration_id"] = configuration_id
            row["parameters_json"] = json.dumps(parameters, sort_keys=True)
            raw_rows.append(row)
            configuration_rows.append(row)
        summary = summarize_configuration(configuration_rows)
        summaries.append(
            {
                "policy": policy_name,
                "configuration_id": configuration_id,
                "parameters": parameters,
                "summary": summary,
            }
        )
        print(
            f"CONFIG={configuration_id} "
            f"FND={summary['median_fnd_or_censor_round']:.1f} "
            f"DELIVERY={summary['median_delivery_ratio']:.4f} "
            f"FAIRNESS={summary['median_queue_fairness']:.4f}",
            flush=True,
        )

    selected = {}
    for policy_name in sorted({row["policy"] for row in summaries}):
        choices = [row for row in summaries if row["policy"] == policy_name]
        winner = max(
            choices,
            key=lambda row: selection_key(policy_name, row["summary"]),
        )
        selected[policy_name] = winner

    run_dir = ROOT / "outputs" / "phase3" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_path = run_dir / "raw_trials.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw_rows[0].keys()))
        writer.writeheader()
        writer.writerows(raw_rows)
    report = {
        "status": "complete",
        "schedule_schema_version": 2,
        "seeds": seeds,
        "protected_seed_overlap": sorted(overlap),
        "horizon": args.horizon,
        "candidate_count": len(summaries),
        "run_count": len(raw_rows),
        "selection_rules": {
            "energy_proportional": "maximize median FND-or-censor, then delivery ratio, then packet fairness",
            "harvest_proportional": "maximize median FND-or-censor, then delivery ratio, then packet fairness",
            "s2a2mac_adapted": "maximize median FND-or-censor, then delivery ratio, then packet fairness",
            "ffss_adapted": "maximize delivery ratio, then packet fairness, then median FND-or-censor, matching FFSS channel-utilization intent",
        },
        "adaptation_limit": (
            "No unpublished S2A2MAC HMM threshold was invented. The sweep only "
            "tests the explicit energy/load mixture used to replace unavailable "
            "source parameters. FFSS remains a round-level feasible-first adaptation."
        ),
        "schedule_metadata": schedule_metadata,
        "configurations": summaries,
        "selected": selected,
    }
    report_path = run_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"BASELINE_TUNING_RUNS={len(raw_rows)}")
    print("BASELINE_TUNING_PASS=True")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())