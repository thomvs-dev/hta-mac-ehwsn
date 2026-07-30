"""Per-round frozen-CH integration for the HTA-MAC environment."""

from __future__ import annotations

import numpy as np

from .intra_cluster_mac_env import IntraClusterMACEnv


class ScheduledIntraClusterMACEnv(IntraClusterMACEnv):
    """Use an externally frozen HEART-CH schedule without retraining it."""

    def reset(self, *, seed: int, frozen_snapshot: dict):
        schedule = frozen_snapshot.get("schedule")
        if not schedule:
            raise ValueError("frozen_snapshot must include a non-empty schedule")
        self.frozen_schedule = schedule
        return super().reset(seed=seed, frozen_snapshot=schedule[0])

    def _install_schedule_round(self, index: int) -> None:
        frame = self.frozen_schedule[min(index, len(self.frozen_schedule) - 1)]
        self.positions = np.asarray(frame["positions"], dtype=np.float64).copy()
        self.cluster_heads = np.asarray(
            frame["cluster_heads"], dtype=np.int64
        ).copy()
        self.embedding = np.asarray(frame["stgcn_embedding"]).copy()
        self._assign_clusters()

    def step(self, action):
        output = super().step(action)
        self._install_schedule_round(self.round)
        state, reward, terminated, truncated, info = output
        return self._state(), reward, terminated, truncated, info
