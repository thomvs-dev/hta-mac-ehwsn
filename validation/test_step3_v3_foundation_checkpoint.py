from __future__ import annotations

from pathlib import Path

from experiments.create_step3_v3_foundation_checkpoint import validate_probe_artifacts


ROOT = Path(__file__).resolve().parents[1]


def test_failed_performance_probe_is_valid_structural_foundation_checkpoint():
    run_dir = ROOT / "outputs" / "phase2" / "step3_bounded_foundation_probe_local"
    if not run_dir.is_dir():
        return
    report = validate_probe_artifacts(run_dir)
    assert report["status"] == "step3_v3_foundation_checkpoint_created"
    assert report["performance_gate_applied"] is False
    assert all(report["gates"].values())
