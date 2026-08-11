"""Step 3 v3 QoS controller candidates with calibrated floors and EMA updates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.qos_constraints import QoSConstraintConfig, QoSConstraintController


@dataclass(frozen=True)
class Step3QoSConstraintConfig(QoSConstraintConfig):
    update_variant: str = "episode_end"
    ema_decay: float = 0.0
    multiplier_floor: dict[str, float] | None = None
    target_penalty_fraction_min: float = 0.02
    target_penalty_fraction_max: float = 0.10

    @classmethod
    def from_payload(cls, payload):
        compatible = dict(payload)
        compatible["status"] = "frozen_development_constraint"
        base = QoSConstraintConfig.from_payload(compatible)
        variant = payload.get("update_variant", "episode_end")
        if variant not in {"episode_end", "ema_episode_end"}:
            raise ValueError("unsupported Step 3 QoS update variant")
        decay = float(payload.get("ema_decay", 0.0))
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must lie in [0,1)")
        floors = {name: float(payload.get("multiplier_floor", {}).get(name, 0.0)) for name in ("delivery", "stale", "fairness")}
        if any(not 0.0 <= value <= base.maximum_multiplier for value in floors.values()):
            raise ValueError("QoS multiplier floor outside configured bounds")
        target = payload.get("target_active_penalty_fraction", {"minimum": 0.02, "maximum": 0.10})
        lower, upper = float(target["minimum"]), float(target["maximum"])
        if not 0.0 < lower <= upper < 0.20:
            raise ValueError("invalid active QoS penalty-fraction target")
        return cls(
            base.minimum_delivery_ratio, base.maximum_stale_drop_ratio,
            base.minimum_queue_fairness, base.learning_rates,
            base.initial_multipliers, base.maximum_multiplier,
            base.update_interval, base.fairness_warmup_steps,
            base.penalty_scale, base.ratio_scope, base.demand_field,
            base.fairness_metric_name, variant, decay, floors, lower, upper,
        )


class Step3QoSConstraintController(QoSConstraintController):
    def __init__(self, config):
        self.ema_signed = {name: 0.0 for name in ("delivery", "stale", "fairness")}
        super().__init__(config)
        self._apply_floors()

    def _apply_floors(self):
        floors = self.config.multiplier_floor or {}
        for name in self.multipliers:
            self.multipliers[name] = max(self.multipliers[name], float(floors.get(name, 0.0)))

    def end_episode(self):
        if self.config.update_variant != "ema_episode_end":
            result = super().end_episode()
            self._apply_floors()
            result["multipliers"] = dict(self.multipliers)
            return result
        updated = False
        if self.steps:
            decay = self.config.ema_decay
            for name in self.multipliers:
                self.ema_signed[name] = decay * self.ema_signed[name] + (1.0 - decay) * self.last_signed[name]
                value = self.multipliers[name] + self.config.learning_rates[name] * self.ema_signed[name]
                floor = float((self.config.multiplier_floor or {}).get(name, 0.0))
                self.multipliers[name] = float(np.clip(value, floor, self.config.maximum_multiplier))
            updated = True
        return {
            "updated": updated,
            "multipliers": dict(self.multipliers),
            "final_signed_violations": dict(self.last_signed),
            "ema_signed_violations": dict(self.ema_signed),
            "update_variant": self.config.update_variant,
        }

    def state_dict(self):
        state = super().state_dict()
        state["ema_signed_violations"] = dict(self.ema_signed)
        state["update_variant"] = self.config.update_variant
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.ema_signed = {name: float(state.get("ema_signed_violations", {}).get(name, 0.0)) for name in self.multipliers}
        self._apply_floors()
