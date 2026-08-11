"""Step 3 v3 environment with observable CH context and QoS audit counters."""

from __future__ import annotations

from envs.step3_lifetime_env import Step3DynamicClusterTrainingEnv
from envs.step3_policy_observation import build_step3_observation, step3_observation_layout


class Step3V3DynamicClusterTrainingEnv(Step3DynamicClusterTrainingEnv):
    def __init__(self, *args, risk_config=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._step3_v3_risk_config = risk_config
        self.step3_qos_counts = None

    def reset(self):
        result = super().reset()
        self.step3_qos_counts = {"delivered": 0, "demand": 0, "stale": 0, "fairness": 1.0}
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
        return observation, mask, done, info
