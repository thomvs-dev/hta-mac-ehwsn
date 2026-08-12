"""Step 3 v3 environment with observable CH context and QoS audit counters."""

from __future__ import annotations

import numpy as np

from envs.step3_lifetime_env import Step3DynamicClusterTrainingEnv
from envs.step3_policy_observation import build_step3_observation, step3_observation_layout


def episode_service_fairness(delivered, offered):
    """Jain fairness over per-node cumulative service ratios.

    Only nodes with positive offered demand enter the cohort. An episode with
    no offered demand is vacuously fair, which avoids a terminal empty-cluster
    artifact while keeping the denominator explicit.
    """
    delivered = np.asarray(delivered, dtype=np.float64)
    offered = np.asarray(offered, dtype=np.float64)
    if delivered.shape != offered.shape:
        raise ValueError("delivered/offered service arrays must have equal shape")
    if np.any(delivered < 0.0) or np.any(offered < 0.0):
        raise ValueError("service arrays must be nonnegative")
    eligible = offered > 0.0
    if not np.any(eligible):
        return 1.0
    ratios = np.clip(delivered[eligible] / offered[eligible], 0.0, 1.0)
    denominator = len(ratios) * np.square(ratios).sum()
    return float(ratios.sum() ** 2 / denominator) if denominator > 0.0 else 0.0


class Step3V3DynamicClusterTrainingEnv(Step3DynamicClusterTrainingEnv):
    def __init__(self, *args, risk_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._step3_v3_risk_config = risk_config
        self.step3_qos_counts = None

    def reset(self):
        result = super().reset()
        self.step3_qos_counts = {"delivered": 0, "demand": 0, "stale": 0, "fairness": 1.0}
        self.step3_episode_offered_per_node = np.zeros(self.base.n_nodes, dtype=np.int64)
        self.step3_episode_delivered_per_node = np.zeros(self.base.n_nodes, dtype=np.int64)
        self.step3_qos_counts["episode_service_fairness"] = 1.0
        return result

    def _observation(self, state):
        if self._step3_v3_risk_config is None:
            from envs.step3_lifetime_env import _ACTIVE_RISK_CONFIG
            risk_config = _ACTIVE_RISK_CONFIG
        else:
            risk_config = self._step3_v3_risk_config
        if risk_config is None:
            raise RuntimeError("Step 3 v3 risk config is not installed")
        return build_step3_observation(
            self.base, state, self._mask(), ch=self.ch, members=self.members,
            risk_config=risk_config,
        )

    @property
    def observation_layout(self):
        return step3_observation_layout(self.base)

    def step(self, member_action):
        observation, mask, done, info = super().step(member_action)
        counts = self.step3_qos_counts
        counts["delivered"] += int(info["target_packets_delivered"])
        counts["demand"] += int(info["target_packets_offered"])
        counts["stale"] += int(info["target_stale_drops"])
        counts["fairness"] = float(info["target_cluster_service_fairness"])
        self.step3_episode_offered_per_node += np.asarray(
            info["target_packets_offered_per_node"], dtype=np.int64
        )
        self.step3_episode_delivered_per_node += np.asarray(
            info["target_packets_delivered_per_node"], dtype=np.int64
        )
        counts["episode_service_fairness"] = episode_service_fairness(
            self.step3_episode_delivered_per_node,
            self.step3_episode_offered_per_node,
        )
        info["target_episode_service_fairness"] = counts["episode_service_fairness"]
        return observation, mask, done, info
