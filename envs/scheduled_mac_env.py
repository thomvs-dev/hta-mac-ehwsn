"""Per-round frozen-CH integration with explicit schedule censoring."""

from __future__ import annotations

import numpy as np

from .intra_cluster_mac_env import IntraClusterMACEnv


class ScheduledIntraClusterMACEnv(IntraClusterMACEnv):
    """Use a shared exogenous HEART-CH schedule without stale-frame replay."""

    def reset(self, *, seed: int, frozen_snapshot: dict):
        schedule = frozen_snapshot.get("schedule")
        if not schedule:
            raise ValueError("frozen_snapshot must include a non-empty schedule")
        self.frozen_schedule = schedule
        self.schedule_metadata = frozen_snapshot.get("schedule_metadata", {})
        self.schedule_exhausted = False
        return super().reset(seed=seed, frozen_snapshot=schedule[0])

    def _install_schedule_round(self, index: int) -> bool:
        if index >= len(self.frozen_schedule):
            self.schedule_exhausted = True
            return False
        frame = self.frozen_schedule[index]
        self.positions = np.asarray(frame["positions"], dtype=np.float64).copy()
        self.cluster_heads = np.asarray(
            frame["cluster_heads"], dtype=np.int64
        ).copy()
        self.embedding = np.asarray(frame["stgcn_embedding"]).copy()
        self._assign_clusters()
        return True

    def step(self, action):
        state, reward, terminated, truncated, info = super().step(action)
        if not terminated and not truncated:
            if self._install_schedule_round(self.round):
                state = self._state()
            else:
                truncated = True
                info["right_censored_schedule"] = True
        info["schedule_coverage_rounds"] = len(self.frozen_schedule)
        info["schedule_exhausted"] = self.schedule_exhausted
        return state, reward, terminated, truncated, info
