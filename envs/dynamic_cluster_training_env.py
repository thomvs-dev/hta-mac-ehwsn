"""Scheduled single-cluster training wrapper for dynamic Phase 2 curriculum."""

from __future__ import annotations

import numpy as np

from .fixed_cluster_training_env import FixedClusterTrainingEnv

from .policy_observation import (
    LEGACY_POLICY_SCHEMA,
    build_policy_observation,
    policy_feature_layout,
)


class DynamicClusterTrainingEnv:
    """Train one cluster rank while the frozen HEART-CH schedule evolves."""

    def __init__(
        self,
        base_env,
        schedule_bundle: dict,
        *,
        seed: int,
        target_rank: int,
        observation_schema: str = LEGACY_POLICY_SCHEMA,
    ):
        self.base = base_env
        self.bundle = schedule_bundle
        self.seed = int(seed)
        self.target_rank = int(target_rank)
        self.observation_schema = str(observation_schema)
        self.target_cluster = None
        self.members = None
        self.ch = None
        self.cumulative_service = None

    def _select_target(self):
        cluster_count = len(self.base.cluster_heads)
        if cluster_count <= 0:
            raise RuntimeError("scheduled frame contains no cluster heads")
        self.target_cluster = self.target_rank % cluster_count
        self.ch = int(self.base.cluster_heads[self.target_cluster])
        self.members = np.flatnonzero(
            (self.base.cluster_of == self.target_cluster)
            & (np.arange(self.base.n_nodes) != self.ch)
        )

    def reset(self):
        state, info = self.base.reset(
            seed=self.seed, frozen_snapshot=self.bundle
        )
        self.cumulative_service = np.zeros(
            self.base.n_nodes, dtype=np.float64
        )
        self._select_target()
        return self._observation(state), self._mask(), info

    @property
    def member_count(self) -> int:
        return int(len(self.members))

    def _observation(self, state):
        return build_policy_observation(
            self.base,
            state,
            self._mask(),
            schema=self.observation_schema,
        )

    @property
    def observation_layout(self):
        return policy_feature_layout(self.base, self.observation_schema)

    def _mask(self):
        mask = np.zeros(self.base.n_nodes, dtype=bool)
        if self.base.alive[self.ch]:
            mask[self.members] = self.base.alive[self.members]
        return mask

    def _trajectory_indicators(self, members):
        original_members = self.members
        self.members = members
        try:
            helper = FixedClusterTrainingEnv._trajectory_indicators
            return helper(self)
        finally:
            self.members = original_members

    @staticmethod
    def _is_done(terminated, truncated, deaths, ch_alive):
        """Do not treat schedule-driven empty membership as node death."""
        return bool(
            terminated or truncated or deaths > 0 or not ch_alive
        )

    def step(self, member_action):
        member_action = np.asarray(member_action, dtype=np.int64)
        if member_action.shape != (self.base.n_nodes,):
            raise ValueError("dynamic action must preserve global node identity")
        current_members = self.members.copy()
        member_mask = np.zeros(self.base.n_nodes, dtype=bool)
        member_mask[current_members] = True
        if np.any(member_action[~member_mask] != 0):
            raise ValueError("non-target branches must receive zero slots")
        target_action = member_action[current_members].copy()
        current_cluster = int(self.target_cluster)
        current_ch = int(self.ch)
        alive_before = self.base.alive[current_members].copy()
        ch_alive_before = bool(self.base.alive[current_ch])
        target_action[~alive_before] = 0
        if not ch_alive_before:
            target_action[:] = 0
        if int(target_action.sum()) > self.base.cfg.frame_slot_budget:
            raise ValueError("target-cluster action exceeds slot budget")

        combined = self.base.static_equal_action()
        combined[current_members] = target_action
        queue_before = self.base.queue[current_members].copy()
        delivered = np.minimum(queue_before, target_action)
        expiring_before = np.asarray(
            [sum(age >= self.base.cfg.packet_ttl_rounds for age in self.base.packet_ages[node])
             for node in current_members],
            dtype=np.int64,
        )
        high, declining = self._trajectory_indicators(current_members)
        alive_count = max(1, int(alive_before.sum()))
        total_before = self.base.total_packets

        next_state, _, terminated, truncated, info = self.base.step(combined)
        trace = info["energy_trace"]
        alive_after = self.base.alive[current_members].copy()
        deaths = FixedClusterTrainingEnv._death_count(
            alive_before,
            alive_after,
            ch_alive_before,
            self.base.alive[current_ch],
        )
        self.cumulative_service[current_members] += delivered

        raw_terms = {
            "packets_delivered": float(delivered.sum() / alive_count),
            "idle_energy_j": -float(trace["idle_energy"][current_members].sum()),
            "deaths": -float(deaths),
            "high_harvest_alignment": float(
                np.sum(high * target_action / self.base.cfg.n_max)
                / alive_count
            ),
            "declining_allocation": -float(
                np.sum(declining * target_action / self.base.cfg.n_max)
                / alive_count
            ),
            "queue_fairness": FixedClusterTrainingEnv._jain(
                self.cumulative_service[current_members]
            ),
        }
        info["reward_raw_terms"] = raw_terms
        info["target_packets_delivered"] = int(delivered.sum())
        # Same-step pre-service demand remains coherent as the target rotates.
        info["target_packets_offered"] = int(queue_before[alive_before].sum())
        info["target_packets_generated"] = int(alive_after.sum())
        info["target_expiring_packets"] = int(expiring_before.sum())
        info["target_stale_drops"] = int(
            np.maximum(0, expiring_before[alive_after] - delivered[alive_after]).sum()
        )
        info["global_packets_delta"] = int(
            self.base.total_packets - total_before
        )
        info["target_cluster"] = current_cluster
        info["target_ch"] = current_ch
        info["target_members"] = current_members.copy()
        info["target_cluster_service_fairness"] = raw_terms["queue_fairness"]

        done = self._is_done(
            terminated,
            truncated,
            deaths,
            self.base.alive[current_ch],
        )
        if len(self.base.cluster_heads):
            self._select_target()
        return self._observation(next_state), self._mask(), done, info