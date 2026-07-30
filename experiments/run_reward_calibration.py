"""Canonical reward calibration on the non-degenerate seed-2100 cluster."""

from __future__ import annotations

import json
from pathlib import Path

import calibrate_reward_scales as calibration


_snapshot = calibration.frozen_initial_snapshot
_fixed_env = calibration.FixedClusterTrainingEnv


def _seed_2100_snapshot(policy, _seed):
    return _snapshot(policy, 2100)


class _Seed2100FixedEnv(_fixed_env):
    def __init__(self, base_env, frozen_snapshot, *, seed=2100):
        super().__init__(base_env, frozen_snapshot, seed=2100)


calibration.frozen_initial_snapshot = _seed_2100_snapshot
calibration.FixedClusterTrainingEnv = _Seed2100FixedEnv


if __name__ == "__main__":
    exit_code = calibration.main()
    root = Path(__file__).resolve().parents[1]
    report_path = root / "outputs" / "logs" / "phase2_reward_calibration.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["seed"] = 2100
    report["calibration_cluster_layout"] = [24, 19, 20, 31, 1]
    report["selection_reason"] = "median-sized non-degenerate 20-member cluster"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(exit_code)
