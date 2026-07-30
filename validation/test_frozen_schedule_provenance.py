from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

from core.ch_selection.frozen_schedule_full import (
    SCHEDULE_SCHEMA_VERSION,
    frozen_ch_schedule_full,
)


class _FakeEnv:
    N = 2
    death_threshold = 0.1

    def __init__(self):
        self.node_positions = np.zeros((2, 2), dtype=np.float64)
        self.hmm_states = np.zeros(2, dtype=np.int64)
        self.thermal_hmm_states = np.zeros(2, dtype=np.int64)
        self._alive = np.ones(2, dtype=bool)
        self._round = 0

    def reset(self, seed):
        return np.zeros((2, 1), dtype=np.float32), {}

    def get_graph_data(self):
        return np.empty((2, 0), dtype=np.int64), np.empty(0)

    def get_alive_mask(self):
        return self._alive.copy()

    def step(self, action):
        self._round += 1
        terminated = self._round == 2
        if terminated:
            self._alive[:] = False
        info = {
            "alive_count": int(self._alive.sum()),
            "episode_stats": {"rounds_survived": self._round},
        }
        return np.zeros((2, 1), dtype=np.float32), 0.0, terminated, False, info


class _NoisyFakePolicy:
    def __init__(self, upstream: Path):
        self.upstream = upstream

    def select(self, state, edge_index, edge_weight, alive_mask):
        action = np.zeros(2, dtype=np.int64)
        alive = np.flatnonzero(alive_mask)
        if alive.size:
            chosen = alive[int(torch.rand(()) * alive.size) % alive.size]
            action[chosen] = 1
        embedding = torch.rand((2, 3)).numpy()
        return action, embedding


def test_full_schedule_is_seed_deterministic_and_reports_upstream_terminal(
    monkeypatch,
):
    train = types.ModuleType("train")
    train.load_stage1_params = lambda path: object()
    train.make_env = lambda params, seed, mode, max_rounds: _FakeEnv()
    config = types.ModuleType("config")
    config.STAGE1_PARAMS_PATH = "unused.mat"
    config.AGENT_MODE = "marl"
    monkeypatch.setitem(sys.modules, "train", train)
    monkeypatch.setitem(sys.modules, "config", config)

    policy = _NoisyFakePolicy(Path("."))
    first = frozen_ch_schedule_full(policy, 73, horizon=10)
    second = frozen_ch_schedule_full(policy, 73, horizon=10)

    assert first["schedule_schema_version"] == SCHEDULE_SCHEMA_VERSION
    assert first["coverage_rounds"] == 2
    assert first["stop_reason"] == "upstream_episode_terminated"
    assert first["upstream_termination_cause"] == "alive_fraction_below_death_threshold"
    assert first["terminal_alive_count"] == 0
    assert first["upstream_episode_stats"]["rounds_survived"] == 2
    for left, right in zip(first["frames"], second["frames"]):
        np.testing.assert_array_equal(left["cluster_heads"], right["cluster_heads"])
        np.testing.assert_allclose(left["stgcn_embedding"], right["stgcn_embedding"])