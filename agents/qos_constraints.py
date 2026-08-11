"""Development-only primal-dual QoS constraint controller for Phase 2D."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QoSConstraintConfig:
    minimum_delivery_ratio: float
    maximum_stale_drop_ratio: float
    minimum_queue_fairness: float
    learning_rates: dict[str, float]
    initial_multipliers: dict[str, float]
    maximum_multiplier: float
    update_interval: str
    fairness_warmup_steps: int
    penalty_scale: float
    ratio_scope: str
    demand_field: str
    fairness_metric_name: str

    @classmethod
    def from_payload(cls, payload: dict) -> "QoSConstraintConfig":
        if payload.get("status") != "frozen_development_constraint":
            raise ValueError("QoS constraints are not frozen development evidence")
        if payload.get("apply_to") != "raw_learning_reward_before_frozen_c51_scale":
            raise ValueError("unsupported QoS constraint application scope")
        if payload.get("physical_metrics_modified") is not False:
            raise ValueError("QoS constraints must not modify physical metrics")
        if payload.get("held_out_seeds_used") is not False:
            raise ValueError("QoS constraints must not use held-out seeds")
        update_interval = payload.get("update_interval")
        if update_interval not in {"environment_step", "episode_end"}:
            raise ValueError("unsupported QoS multiplier update interval")
        ratio_scope = payload.get("ratio_scope")
        scopes = {
            "episode_cumulative_target_cluster": "target_packets_generated",
            "episode_cumulative_target_backlog_service": "target_packets_offered",
        }
        if ratio_scope not in scopes:
            raise ValueError("unsupported QoS ratio scope")
        demand_field = payload.get("demand_field", scopes[ratio_scope])
        if demand_field != scopes[ratio_scope]:
            raise ValueError("QoS demand field does not match ratio scope")
        fairness_metric_name = payload.get(
            "fairness_metric_name", "target_cluster_service_fairness"
        )
        if fairness_metric_name != "target_cluster_service_fairness":
            raise ValueError("unsupported QoS fairness metric")

        thresholds = (
            float(payload["minimum_delivery_ratio"]),
            float(payload["maximum_stale_drop_ratio"]),
            float(payload["minimum_queue_fairness"]),
        )
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in thresholds):
            raise ValueError("QoS thresholds must be finite probabilities")
        names = ("delivery", "stale", "fairness")
        learning_rates = {name: float(payload["multiplier_learning_rate"][name]) for name in names}
        initial = {name: float(payload["initial_multiplier"][name]) for name in names}
        maximum = float(payload["maximum_multiplier"])
        if not np.isfinite(maximum) or maximum <= 0.0:
            raise ValueError("maximum multiplier must be finite and positive")
        if any(not np.isfinite(value) or value < 0.0 for value in learning_rates.values()):
            raise ValueError("multiplier learning rates must be finite and nonnegative")
        if any(not np.isfinite(value) or not 0.0 <= value <= maximum for value in initial.values()):
            raise ValueError("initial multipliers must lie within the configured bounds")
        warmup = int(payload.get("fairness_warmup_steps", 0))
        penalty_scale = float(payload.get("penalty_scale", 1.0))
        if warmup < 0:
            raise ValueError("fairness warmup must be nonnegative")
        if not np.isfinite(penalty_scale) or not 0.0 < penalty_scale <= 1.0:
            raise ValueError("penalty scale must be finite and in (0, 1]")
        return cls(
            *thresholds, learning_rates, initial, maximum,
            update_interval, warmup, penalty_scale,
            ratio_scope, demand_field, fairness_metric_name
        )


class QoSConstraintController:
    """Apply nonnegative Lagrangian penalties using cumulative episode QoS."""

    def __init__(self, config: QoSConstraintConfig):
        self.config = config
        self.multipliers = dict(config.initial_multipliers)
        self.begin_episode()

    def begin_episode(self) -> None:
        self.delivered = 0
        self.demand = 0
        self.last_fairness = 1.0
        self.last_signed = {name: 0.0 for name in self.multipliers}
        self.stale = 0
        self.steps = 0

    def evaluate(
        self, *, delivered: int, stale: int, fairness: float,
        generated: int | None = None, offered: int | None = None,
    ):
        supplied = {
            "target_packets_generated": generated,
            "target_packets_offered": offered,
        }
        demand_value = supplied[self.config.demand_field]
        if demand_value is None:
            raise ValueError(f"missing QoS demand count: {self.config.demand_field}")
        counts = (int(delivered), int(demand_value), int(stale))
        if any(value < 0 for value in counts):
            raise ValueError("QoS packet counts must be nonnegative")
        if self.config.demand_field == "target_packets_offered" and counts[0] > counts[1]:
            raise ValueError("delivered packets cannot exceed the same-step offered backlog")
        fairness = float(fairness)
        if not np.isfinite(fairness) or not 0.0 <= fairness <= 1.0:
            raise ValueError("QoS fairness must be finite and in [0, 1]")
        self.last_fairness = fairness
        self.delivered += counts[0]
        self.demand += counts[1]
        self.stale += counts[2]
        self.steps += 1
        denominator = max(1, self.demand)
        delivery_ratio = self.delivered / denominator
        stale_ratio = self.stale / denominator
        legacy_clip = self.config.demand_field == "target_packets_generated"
        ratios = {
            "delivery": min(1.0, delivery_ratio) if legacy_clip else delivery_ratio,
            "stale": min(1.0, stale_ratio) if legacy_clip else stale_ratio,
            "fairness": fairness,
        }
        signed = {
            "delivery": self.config.minimum_delivery_ratio - ratios["delivery"],
            "stale": ratios["stale"] - self.config.maximum_stale_drop_ratio,
            "fairness": self.config.minimum_queue_fairness - ratios["fairness"],
        }
        if self.steps <= self.config.fairness_warmup_steps:
            signed["fairness"] = 0.0
        self.last_signed = dict(signed)
        positive = {name: max(0.0, value) for name, value in signed.items()}
        multiplier_updated = self.config.update_interval == "environment_step"
        if multiplier_updated:
            for name in self.multipliers:
                updated = self.multipliers[name] + self.config.learning_rates[name] * signed[name]
                self.multipliers[name] = float(np.clip(updated, 0.0, self.config.maximum_multiplier))
        penalty_terms = {
            name: -self.config.penalty_scale * self.multipliers[name] * positive[name] for name in positive
        }
        penalty = float(sum(penalty_terms.values()))
        return penalty, {
            "ratios": ratios,
            "signed_violations": signed,
            "positive_violations": positive,
            "multipliers": dict(self.multipliers),
            "penalty_terms": penalty_terms,
            "penalty": penalty,
            "multiplier_update_interval": self.config.update_interval,
            "multiplier_updated": multiplier_updated,
            "cumulative_counts": {
                "delivered": self.delivered,
                "demand": self.demand,
                "stale": self.stale,
            },
            "metric_contract": {
                "ratio_scope": self.config.ratio_scope,
                "demand_field": self.config.demand_field,
                "fairness_metric_name": self.config.fairness_metric_name,
            },
        }

    def evaluate_info(self, info: dict):
        """Evaluate one environment info record under the frozen metric contract."""
        kwargs = {
            "delivered": info["target_packets_delivered"],
            "stale": info["target_stale_drops"],
            "fairness": info["target_cluster_service_fairness"],
        }
        if self.config.demand_field == "target_packets_offered":
            kwargs["offered"] = info["target_packets_offered"]
        else:
            kwargs["generated"] = info["target_packets_generated"]
        return self.evaluate(**kwargs)

    def end_episode(self) -> dict:
        """Update shared multipliers once from the final episode ratios."""
        updated = False
        if self.steps and self.config.update_interval == "episode_end":
            for name in self.multipliers:
                value = self.multipliers[name] + self.config.learning_rates[name] * self.last_signed[name]
                self.multipliers[name] = float(np.clip(value, 0.0, self.config.maximum_multiplier))
            updated = True
        return {
            "updated": updated,
            "multipliers": dict(self.multipliers),
            "final_signed_violations": dict(self.last_signed),
        }
    def state_dict(self) -> dict:
        demand_key = "generated" if self.config.demand_field == "target_packets_generated" else "demand"
        return {
            "multipliers": dict(self.multipliers),
            "episode_counts": {
                "delivered": self.delivered,
                demand_key: self.demand,
                "stale": self.stale,
                "steps": self.steps,
            },
            "metric_contract": {
                "ratio_scope": self.config.ratio_scope,
                "demand_field": self.config.demand_field,
                "fairness_metric_name": self.config.fairness_metric_name,
            },
        }

    def load_state_dict(self, state: dict) -> None:
        multipliers = {name: float(state["multipliers"][name]) for name in self.multipliers}
        if any(not np.isfinite(value) or not 0.0 <= value <= self.config.maximum_multiplier for value in multipliers.values()):
            raise ValueError("invalid saved QoS multiplier state")
        counts = state.get("episode_counts", {})
        self.multipliers = multipliers
        self.delivered = int(counts.get("delivered", 0))
        self.demand = int(counts.get("demand", counts.get("generated", 0)))
        self.stale = int(counts.get("stale", 0))
        self.steps = int(counts.get("steps", 0))
        if min(self.delivered, self.demand, self.stale, self.steps) < 0:
            raise ValueError("invalid saved QoS episode counts")
