"""Literature-grounded RSG/deadline/energy teacher for HTA-MAC V2 Phase B3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.guarded_marginal_teacher import (
    GuardedTeacherConfig,
    _packet_costs,
    _repair_fairness,
    _repair_stale,
    _required_expiring_service,
    _served_expiring,
)
from envs.step3_v3_env import episode_service_fairness


@dataclass(frozen=True)
class RSGEnergyConfig:
    energy_weight: float
    deadline_weight: float
    tsls_weight: float
    harvest_credit: float
    queue_weight: float = 1.0
    delivery_debt_weight: float = 1.0
    depletion_weight: float = 1.0
    stale_guard: float = 0.035
    fairness_guard: float = 0.92
    reserve_floor: float = 0.08
    delivery_floor: float = 0.45

    @classmethod
    def from_payload(cls, payload: dict) -> "RSGEnergyConfig":
        values = {name: float(payload[name]) for name in cls.__dataclass_fields__}
        result = cls(**values)
        vector = np.asarray(list(values.values()), dtype=np.float64)
        if np.any(~np.isfinite(vector)) or np.any(vector < 0.0):
            raise ValueError("RSG-energy coefficients must be finite and nonnegative")
        for name in ("stale_guard", "fairness_guard", "reserve_floor", "delivery_floor"):
            if not 0.0 <= getattr(result, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")
        return result


class RSGEnergyTeacher:
    """Stateful packet scheduler with time-since-last-service (TSLS)."""

    def __init__(self, n_nodes: int, config: RSGEnergyConfig):
        self.config = config
        self.tsls = np.zeros(int(n_nodes), dtype=np.int64)
        self.round_index = 0

    def _projected_fairness(self, env, action: np.ndarray) -> float:
        offered = env.step3_episode_offered_per_node.astype(np.float64) + env.base.queue
        delivered = env.step3_episode_delivered_per_node.astype(np.float64) + action
        return episode_service_fairness(delivered, offered)

    def _scores(self, env, mask, caps, action, costs) -> np.ndarray:
        cfg, base = self.config, env.base
        eligible = mask & (action < caps)
        score = np.full(base.n_nodes, -np.inf, dtype=np.float64)
        if not np.any(eligible):
            return score
        cost_scale = max(float(np.median(costs[eligible])), 1e-12)
        forecast = np.maximum(base._state()[:, 1].astype(np.float64), 0.0)
        forecast_scale = max(float(forecast[mask].max(initial=0.0)), 1e-12)
        initial = max(float(base.cfg.initial_energy_j), 1e-12)
        offered = env.step3_episode_offered_per_node.astype(np.float64) + base.queue
        delivered = env.step3_episode_delivered_per_node.astype(np.float64) + action
        service_ratio = np.divide(delivered, np.maximum(offered, 1.0))
        debt = np.maximum(0.0, cfg.delivery_floor - service_ratio)
        ttl_scale = max(1.0, float(base.cfg.packet_ttl_rounds))
        queue_scale = max(1.0, float(base.cfg.queue_max_packets))
        tsls_scale = ttl_scale + 1.0
        for node in np.flatnonzero(eligible):
            layer = int(action[node])
            age = float(base.packet_ages[node][layer])
            queue_pressure = max(0.0, float(caps[node] - action[node])) / queue_scale
            deadline_pressure = (age / ttl_scale) ** 2
            regularity = min(float(self.tsls[node]) / tsls_scale, 3.0)
            residual = float(np.clip(base.energy[node] / initial, 0.0, 1.0))
            spill_credit = float(forecast[node] / forecast_scale) * residual
            urgency = (
                cfg.queue_weight * queue_pressure
                + cfg.deadline_weight * deadline_pressure
                + cfg.tsls_weight * regularity
                + cfg.delivery_debt_weight * debt[node]
            )
            # Queue/battery complementarity: urgent packets get more value when
            # energy is currently usable; forecast at a fuller battery creates
            # an additional transmit-now credit to avoid harvest spill.
            complementarity = urgency * (0.5 + 0.5 * residual)
            score[node] = (
                complementarity + cfg.harvest_credit * spill_credit
                - cfg.energy_weight * float(costs[node]) / cost_scale
            )
        return score

    def action(self, env, mask, caps, budget: int, *, return_audit: bool = False):
        mask = np.asarray(mask, dtype=bool)
        caps = np.asarray(caps, dtype=np.int64)
        target = min(int(budget), int(caps[mask].sum()))
        action = np.zeros_like(caps, dtype=np.int64)
        if target > 0:
            guard_cfg = GuardedTeacherConfig(
                energy_weight=self.config.energy_weight,
                depletion_weight=self.config.depletion_weight,
                fairness_guard=self.config.fairness_guard,
                stale_guard=self.config.stale_guard,
                reserve_floor=self.config.reserve_floor,
            )
            costs = _packet_costs(env, guard_cfg)
            for _ in range(target):
                scores = self._scores(env, mask, caps, action, costs)
                eligible = np.flatnonzero(np.isfinite(scores))
                if not len(eligible):
                    break
                node = min(
                    (int(value) for value in eligible),
                    key=lambda value: (
                        -float(scores[value]), int(action[value]),
                        tuple(float(x) for x in env.base.positions[value]),
                    ),
                )
                action[node] += 1
            required = _required_expiring_service(env, guard_cfg)
            action, stale_swaps = _repair_stale(env, action, caps, mask, costs, required)
            action, fairness_swaps = _repair_fairness(
                env, action, caps, mask, costs, self.config.fairness_guard, required
            )
        else:
            required, stale_swaps, fairness_swaps = 0, [], []
        feasible = bool(
            np.all(action >= 0) and np.all(action <= caps) and np.all(action[~mask] == 0)
            and int(action.sum()) == target
        )
        if not feasible:
            raise RuntimeError("RSG-energy teacher violated feasible-service projection")
        self.tsls[np.asarray(env.base.alive, dtype=bool)] += 1
        self.tsls[action > 0] = 0
        self.round_index += 1
        if return_audit:
            fairness = self._projected_fairness(env, action)
            served_expiring = _served_expiring(env, action)
            return action, {
                "target_slots": target,
                "required_expiring_service": required,
                "served_expiring": served_expiring,
                "stale_guard_satisfied": served_expiring >= required,
                "stale_swaps": len(stale_swaps),
                "fairness_swaps": len(fairness_swaps),
                "projected_fairness": fairness,
                "fairness_guard_satisfied": target == 0 or fairness + 1e-15 >= self.config.fairness_guard,
                "max_tsls": int(self.tsls.max(initial=0)),
            }
        return action
