"""Run the Phase 1 gate with its resolved structural configuration."""

from __future__ import annotations

from pathlib import Path

import phase1_gate


_original_loader = phase1_gate.load_simple_yaml


def _gate_loader(path):
    path = Path(path)
    if path.name == "phase1.yaml":
        path = path.with_name("phase1_gate.yaml")
    return _original_loader(path)


phase1_gate.load_simple_yaml = _gate_loader

if __name__ == "__main__":
    raise SystemExit(phase1_gate.main())
