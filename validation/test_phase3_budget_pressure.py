"""Tests for Phase 3 allocation pressure and B-utilization accounting."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from experiments.run_phase3_pilot import allocation_pressure


def test_allocation_pressure_distinguishes_demand_contention_from_budget_binding():
    env = SimpleNamespace(
        n_nodes=6,
        cluster_heads=np.asarray([0, 3]),
        cluster_of=np.asarray([0, 0, 0, 1, 1, 1]),
        alive=np.ones(6, dtype=bool),
        queue=np.asarray([0, 3, 3, 0, 1, 0]),
        cfg=SimpleNamespace(n_max=3, frame_slot_budget=4),
    )
    pressure = allocation_pressure(env, np.asarray([0, 2, 1, 0, 3, 0]))
    assert pressure == {
        "allocated_slots": 6,
        "feasible_demand_slots": 7,
        "active_clusters": 2,
        "binding_clusters": 0,
        "contended_clusters": 1,
        "available_budget_slots": 8,
    }


def test_dead_cluster_head_removes_cluster_from_pressure_denominator():
    env = SimpleNamespace(
        n_nodes=3,
        cluster_heads=np.asarray([0]),
        cluster_of=np.zeros(3, dtype=np.int64),
        alive=np.asarray([False, True, True]),
        queue=np.asarray([0, 3, 3]),
        cfg=SimpleNamespace(n_max=3, frame_slot_budget=4),
    )
    pressure = allocation_pressure(env, np.zeros(3, dtype=np.int64))
    assert pressure["active_clusters"] == 0
    assert pressure["available_budget_slots"] == 0
