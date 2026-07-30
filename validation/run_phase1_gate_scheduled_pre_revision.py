"""Run the definitive Phase 1 gate with per-round frozen CH decisions."""

from __future__ import annotations

import json
from pathlib import Path

import phase1_gate
from core.ch_selection.frozen_schedule import frozen_ch_schedule
from envs.scheduled_mac_env import ScheduledIntraClusterMACEnv


_original_loader = phase1_gate.load_simple_yaml


def _gate_loader(path):
    path = Path(path)
    if path.name == "phase1.yaml":
        path = path.with_name("phase1_gate.yaml")
    return _original_loader(path)


def _schedule_bundle(policy, seed):
    schedule = frozen_ch_schedule(policy, seed, max_rounds=2000)
    bundle = dict(schedule[0])
    bundle["schedule"] = schedule
    return bundle


phase1_gate.load_simple_yaml = _gate_loader
phase1_gate.frozen_initial_snapshot = _schedule_bundle
phase1_gate.IntraClusterMACEnv = ScheduledIntraClusterMACEnv


if __name__ == "__main__":
    exit_code = phase1_gate.main()
    report_path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "logs"
        / "phase1_gate.json"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["cluster_integration"] = {
        "mode": "per_round_schedule_from_frozen_heart_ch_checkpoint",
        "shared_paired_schedule": True,
        "ch_retraining": False,
        "routing_changes": False,
    }
    report["idle_ablation"]["scope"] = (
        "full member and CH Tx/Rx/aggregation/idle accounting"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(exit_code)
