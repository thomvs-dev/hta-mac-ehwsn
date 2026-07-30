"""Run Phase 1 with explicit member-side idle isolation for the ablation."""

from __future__ import annotations

from pathlib import Path

import phase1_gate
from envs.idle_isolation_env import IdleIsolationMACEnv
from envs.intra_cluster_mac_env import IntraClusterMACEnv


_original_loader = phase1_gate.load_simple_yaml


def _gate_loader(path):
    path = Path(path)
    if path.name == "phase1.yaml":
        path = path.with_name("phase1_gate.yaml")
    return _original_loader(path)


class _GateEnvironment(IntraClusterMACEnv):
    """Route only explicit on/off constructions to the isolation variant."""

    def __new__(cls, config, radio, solar, thermal, **kwargs):
        if "idle_energy_enabled" in kwargs:
            return IdleIsolationMACEnv(config, radio, solar, thermal, **kwargs)
        return super().__new__(cls)


phase1_gate.load_simple_yaml = _gate_loader
phase1_gate.IntraClusterMACEnv = _GateEnvironment

if __name__ == "__main__":
    raise SystemExit(phase1_gate.main())
