"""Audit whether the Phase 2B death reward is activated during training."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


def audit_run(run_dir: Path) -> dict[str, object]:
    episodes_path = run_dir / "episodes.jsonl"
    summary_path = run_dir / "summary.json"
    rows = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed_match = re.search(r"seed(\d+)", run_dir.name)
    if not seed_match:
        raise ValueError(f"Cannot infer training seed from {run_dir.name}")

    deaths = [float(-row["raw_terms"]["deaths"]) for row in rows]
    weighted_deaths = [float(-row["weighted_terms"]["deaths"]) for row in rows]
    positive_indices = [index for index, value in enumerate(deaths) if value > 0]
    tail = deaths[-50:]
    death_weight = float(summary["reward_calibration"]["weights"]["deaths"])
    total_steps = sum(int(row["steps"]) for row in rows)
    firing_steps = len(positive_indices)
    weighted_packet_total = sum(
        float(row["weighted_terms"]["packets_delivered"]) for row in rows
    )
    mean_death_at_firing_step = (
        sum(weighted_deaths) / firing_steps if firing_steps else 0.0
    )
    mean_packet_per_logged_step = weighted_packet_total / total_steps

    return {
        "training_seed": int(seed_match.group(1)),
        "run": run_dir.name,
        "episodes": len(rows),
        "death_weight": death_weight,
        "death_events": int(sum(deaths)),
        "weighted_death_magnitude": float(sum(weighted_deaths)),
        "total_logged_steps": total_steps,
        "death_firing_steps": firing_steps,
        "death_firing_step_fraction": firing_steps / total_steps,
        "mean_weighted_death_at_firing_step": mean_death_at_firing_step,
        "mean_weighted_packet_per_logged_step": mean_packet_per_logged_step,
        "death_to_typical_packet_magnitude_ratio": (
            mean_death_at_firing_step / mean_packet_per_logged_step
        ),
        "episodes_with_death": len(positive_indices),
        "episodes_with_death_fraction": len(positive_indices) / len(rows),
        "mean_deaths_per_episode": statistics.fmean(deaths),
        "median_deaths_per_episode": statistics.median(deaths),
        "max_deaths_in_episode": int(max(deaths)),
        "first_death_episode": int(rows[positive_indices[0]]["episode"]) if positive_indices else None,
        "episodes_with_t_fnd": sum(row.get("t_fnd") is not None for row in rows),
        "last_50_death_events": int(sum(tail)),
        "last_50_episodes_with_death": sum(value > 0 for value in tail),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    runs_dir = args.result_root / "runs"
    run_dirs = sorted(path for path in runs_dir.glob("phase2b_confirm_*_125ep") if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No Phase 2B confirmation runs found under {runs_dir}")

    records = [audit_run(path) for path in run_dirs]
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.write_text(json.dumps({"result_root": str(args.result_root), "runs": records}, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps(records, indent=2))
    print(f"WROTE {json_path}")
    print(f"WROTE {csv_path}")


if __name__ == "__main__":
    main()