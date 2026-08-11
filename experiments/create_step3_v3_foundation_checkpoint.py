"""Create a bounded v3 checkpoint for structural audit, not performance gating."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.train_step3_v3_probe as probe
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA


def validate_probe_artifacts(run_dir: Path) -> dict:
    checkpoint_path = run_dir / "branching_c51.pt"
    episodes_path = run_dir / "episodes.jsonl"
    if not checkpoint_path.is_file() or not episodes_path.is_file():
        raise RuntimeError("foundation probe did not persist checkpoint and episode log")
    rows = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    gates = {
        "exactly_one_logged_episode": len(rows) == 1 and int(rows[0].get("episode", -1)) == 1,
        "schema_is_step3_v3": config.get("state_schema") == STEP3_CH_CONTEXT_SCHEMA,
        "input_dimension_is_65": int(config.get("input_dim", -1)) == 65,
        "embedding_start_is_33": int(config.get("embedding_start_dim", -1)) == 33,
        "equivariant_architecture": config.get("architecture") == "equivariant_set_branching",
        "projection_budget_is_16": int(config.get("budget", -1)) == 16,
    }
    if not all(gates.values()):
        raise RuntimeError(f"invalid foundation checkpoint: {gates}")
    return {
        "schema_version": 1,
        "status": "step3_v3_foundation_checkpoint_created",
        "performance_gate_applied": False,
        "reason": "An untrained one-episode checkpoint is used only for same-platform architecture and permutation audits.",
        "gates": gates,
        "checkpoint": str(checkpoint_path),
    }


def main():
    original = list(sys.argv)
    if "--run-name" not in original:
        raise ValueError("--run-name is required")
    run_name = original[original.index("--run-name") + 1]
    training_exit = probe.main()
    report = validate_probe_artifacts(ROOT / "outputs" / "phase2" / run_name)
    report["underlying_training_exit_code"] = int(training_exit)
    report_path = ROOT / "outputs" / "phase2" / run_name / "foundation_checkpoint_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
