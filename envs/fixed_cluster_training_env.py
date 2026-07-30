"""Deterministic single-cluster wrapper for the Phase 2 sanity gate."""

from __future__ import annotations

import numpy as np



SOLAR_HIGH_HARVEST_STATES = (5, 7)  # HEART-CH S6 and S8
SOLAR_DECLINING_STATES = (0, 3, 6)  # HEART-CH S1, S4, and S7


class FixedClusterTrainingEnv:
    """Train one MAC branch set while other clusters use static equal TDMA."""

    def __init__(
        self,
        base_env,
        frozen_snapshot: dict,
        *,
        seed: int = 3210,
        target_cluster: int | None = None,
    ):
        self.base = base_env
        self.snapshot = frozen_snapshot
        self.seed = int(seed)
        self.requested_cluster = target_cluster
        self.target_cluster = None
        self.members = None
        self.ch = None
        self.cumulative_service = None

    def reset(self):
        state, info = self.base.reset(
            seed=self.seed, frozen_snapshot=self.snapshot
        )
        counts = np.bincount(
            self.base.cluster_of, minlength=len(self.base.cluster_heads)
        ) - 1
        if self.requested_cluster is None:
            median = np.median(counts)
            self.target_cluster = int(np.argmin(np.abs(counts - median)))
        else:
            requested = int(self.requested_cluster)
            if requested < 0 or requested >= len(counts):
                raise ValueError("target cluster index outside frozen snapshot" )
            if counts[requested] <= 0:
                raise ValueError("target cluster has no member branches" )
            self.target_cluster = requested
        self.ch = int(self.base.cluster_heads[self.target_cluster])
        self.members = np.flatnonzero(
            (self.base.cluster_of == self.target_cluster)
            & (np.arange(self.base.n_nodes) != self.ch)
        )
        self.cumulative_service = np.zeros(len(self.members), dtype=np.float64)
        return self._observation(state), self._mask(), info

    @property
    def member_count(self) -> int:
        return int(len(self.members))

    def _observation(self, state):
        embedding = np.asarray(self.base.embedding, dtype=np.float32)
        return np.concatenate(
            (state[self.members], embedding[self.members]), axis=1
        ).astype(np.float32)

    def _mask(self):
        return self.base.alive[self.members].copy()

    @staticmethod
    def _death_count(member_before, member_after, ch_before, ch_after):
        """Count newly dead members and the target CH exactly once."""
        member_deaths = np.count_nonzero(
            np.asarray(member_before, dtype=bool)
            & ~np.asarray(member_after, dtype=bool)
        )
        ch_death = int(bool(ch_before) and not bool(ch_after))
        return int(member_deaths + ch_death)

    @staticmethod
    def _is_done(terminated, truncated, deaths, ch_alive, any_member_alive):
        """End the local episode at its first death to align with T_FND."""
        return bool(
            terminated
            or truncated
            or deaths > 0
            or not ch_alive
            or not any_member_alive
        )

    @staticmethod
    def _jain(values):
        values = np.asarray(values, dtype=np.float64)
        denominator = len(values) * np.square(values).sum()
        if denominator <= 0.0:
            return 0.0
        return float(values.sum() ** 2 / denominator)

    def _trajectory_indicators(self):
        """Map solar states to HEART-CH's published suitability taxonomy.

        S6/S8 are rewarded as high-harvest states; S1/S4/S7 are penalized as
        avoid/critical declining states. The synthetic thermal auxiliary has no
        validated state taxonomy, so no categorical thermal label is invented.
        Thermal trajectory information remains present in the observation's
        forecast moments and state-conditioned transition probabilities.
        """
        solar_state = self.base.solar_states[self.members]
        high = np.isin(solar_state, SOLAR_HIGH_HARVEST_STATES)
        declining = np.isin(solar_state, SOLAR_DECLINING_STATES)
        return high, declining

    def step(self, member_action):
        member_action = np.asarray(member_action, dtype=np.int64)
        if member_action.shape != (self.member_count,):
            raise ValueError("member action shape mismatch")
        member_action = member_action.copy()
        alive_before = self._mask()
        ch_alive_before = bool(self.base.alive[self.ch])
        member_action[~alive_before] = 0
        if int(member_action.sum()) > self.base.cfg.frame_slot_budget:
            raise ValueError("target-cluster action exceeds slot budget")

        combined = self.base.static_equal_action()
        combined[self.members] = member_action
        queue_before = self.base.queue[self.members].copy()
        delivered = np.minimum(queue_before, member_action)
        high, declining = self._trajectory_indicators()
        alive_count = max(1, int(alive_before.sum()))
        total_before = self.base.total_packets

        next_state, _, terminated, truncated, info = self.base.step(combined)
        trace = info["energy_trace"]
        alive_after = self._mask()
        deaths = self._death_count(
            alive_before,
            alive_after,
            ch_alive_before,
            self.base.alive[self.ch],
        )
        self.cumulative_service += delivered

        raw_terms = {
            "packets_delivered": float(delivered.sum() / alive_count),
            "idle_energy_j": -float(trace["idle_energy"][self.members].sum()),
            "deaths": -float(deaths),
            "high_harvest_alignment": float(
                np.sum(high * member_action / self.base.cfg.n_max) / alive_count
            ),
            "declining_allocation": -float(
                np.sum(declining * member_action / self.base.cfg.n_max)
                / alive_count
            ),
            "queue_fairness": self._jain(self.cumulative_service),
        }
        info["reward_raw_terms"] = raw_terms
        info["target_packets_delivered"] = int(delivered.sum())
        info["global_packets_delta"] = int(
            self.base.total_packets - total_before
        )
        info["target_cluster"] = self.target_cluster
        info["target_ch"] = self.ch
        done = self._is_done(
            terminated,
            truncated,
            deaths,
            self.base.alive[self.ch],
            np.any(alive_after),
        )
        return self._observation(next_state), self._mask(), done, info
