"""Measure frozen HEART-CH cluster sizes and derive a primary slot budget."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ch_selection.frozen_heart_ch import FrozenHeartCH
from core.ch_selection.frozen_schedule_full import frozen_ch_schedule_full
from core.configuration import load_simple_yaml


def main() -> int:
    manifest = load_simple_yaml(ROOT / "core" / "frozen_assets.yaml")
    policy = FrozenHeartCH(
        ROOT.parent / "final_repo",
        manifest["checkpoint"]["path"],
        manifest["checkpoint"]["sha256"],
    )
    seeds = [2100, 2101, 2102, 2103, 2104]
    member_counts: list[int] = []
    schedules = []
    for seed in seeds:
        result = frozen_ch_schedule_full(policy, seed, horizon=3000)
        schedules.append(
            {
                "seed": seed,
                "coverage_rounds": result["coverage_rounds"],
                "complete": result["complete"],
                "stop_reason": result["stop_reason"],
            }
        )
        for frame in result["frames"]:
            positions = frame["positions"]
            heads = frame["cluster_heads"]
            distances = np.linalg.norm(
                positions[:, None, :] - positions[heads][None, :, :], axis=2
            )
            assignment = np.argmin(distances, axis=1)
            for local, head in enumerate(heads):
                assignment[head] = local
            counts = np.bincount(assignment, minlength=len(heads)) - 1
            member_counts.extend(int(value) for value in counts)

    values = np.asarray(member_counts, dtype=np.int64)
    median = float(np.median(values))
    primary_t = int(np.ceil(1.3 * median))
    sweep = sorted(
        set(
            max(1, int(round(factor * median)))
            for factor in (1.0, 1.2, 1.3, 1.5)
        )
    )
    report = {
        "seeds": seeds,
        "schedule_coverage": schedules,
        "cluster_member_observations": int(values.size),
        "median_cluster_members": median,
        "iqr_cluster_members": float(
            np.percentile(values, 75) - np.percentile(values, 25)
        ),
        "p90_cluster_members": float(np.percentile(values, 90)),
        "max_cluster_members": int(values.max()),
        "primary_rule": "ceil(1.3 * median_cluster_members)",
        "primary_frame_slot_budget": primary_t,
        "phase5_t_sweep": sweep,
    }
    output = ROOT / "outputs" / "logs" / "cluster_contention_analysis.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"MEDIAN_CLUSTER_MEMBERS={median:.1f}")
    print(f"CLUSTER_MEMBER_IQR={report['iqr_cluster_members']:.1f}")
    print(f"PRIMARY_T={primary_t}")
    print(f"T_SWEEP={sweep}")
    print(f"report={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
