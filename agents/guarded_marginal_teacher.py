"""Guarded deterministic marginal service/energy teacher for HTA-MAC V2.

The teacher first builds a full, feasible energy/depletion allocation. It then
repairs that allocation with the fewest one-packet exchanges it can find to
meet the predeclared stale and episode-service-fairness control boundaries.
The HEART cluster-head schedule remains exogenous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from envs.step3_v3_env import episode_service_fairness


@dataclass(frozen=True)
class GuardedTeacherConfig:
    energy_weight: float
    depletion_weight: float
    fairness_guard: float
    stale_guard: float = 0.033
    reserve_floor: float = 0.08

    @classmethod
    def from_payload(cls, payload: dict) -> "GuardedTeacherConfig":
        values = {name: float(payload[name]) for name in cls.__dataclass_fields__}
        result = cls(**values)
        vector = np.asarray(list(values.values()), dtype=np.float64)
        if np.any(~np.isfinite(vector)) or np.any(vector < 0.0):
            raise ValueError("teacher coefficients must be finite and nonnegative")
        if not 0.0 <= result.stale_guard <= 1.0:
            raise ValueError("stale guard must lie in [0,1]")
        if not 0.0 <= result.fairness_guard <= 1.0:
            raise ValueError("fairness guard must lie in [0,1]")
        if result.reserve_floor > 1.0:
            raise ValueError("reserve floor must lie in [0,1]")
        return result


def _position_key(env, node: int) -> tuple[float, float]:
    position = env.base.positions[int(node)]
    return float(position[0]), float(position[1])


def _packet_costs(env, config: GuardedTeacherConfig) -> np.ndarray:
    """Return permutation-equivariant per-packet energy/depletion costs."""
    base = env.base
    initial = max(float(base.cfg.initial_energy_j), 1e-12)
    ch = int(env.ch)
    ch_packet = float(base.radio.rx(base.cfg.packet_bits) + base.radio.aggregate(base.cfg.packet_bits))
    costs = np.full(base.n_nodes, np.inf, dtype=np.float64)
    raw_energy = np.full(base.n_nodes, np.nan, dtype=np.float64)
    raw_depletion = np.full(base.n_nodes, np.nan, dtype=np.float64)
    for node in np.asarray(env.members, dtype=np.int64):
        member_tx = float(base.radio.tx(
            base.cfg.packet_bits,
            float(np.linalg.norm(base.positions[node] - base.positions[ch])),
        ))
        usable = max(float(base.energy[node]) - config.reserve_floor * initial, 1e-12)
        raw_energy[node] = member_tx + ch_packet
        raw_depletion[node] = member_tx / usable
    valid = np.isfinite(raw_energy)
    energy_scale = max(float(np.median(raw_energy[valid])), 1e-12)
    depletion_scale = max(float(np.median(raw_depletion[valid])), 1e-12)
    costs[valid] = (
        config.energy_weight * raw_energy[valid] / energy_scale
        + config.depletion_weight * raw_depletion[valid] / depletion_scale
    )
    if np.any(~np.isfinite(costs[valid])):
        raise RuntimeError("teacher produced a non-finite marginal cost")
    return costs


def _base_allocation(env, mask, caps, target: int, costs: np.ndarray) -> np.ndarray:
    action = np.zeros_like(caps, dtype=np.int64)
    for _ in range(target):
        eligible = np.flatnonzero(mask & (action < caps))
        if not len(eligible):
            break
        winner = min(
            (int(node) for node in eligible),
            key=lambda node: (float(costs[node]), int(action[node]), _position_key(env, node)),
        )
        action[winner] += 1
    return action


def _served_expiring(env, action: np.ndarray) -> int:
    return sum(
        int(env.base.packet_ages[node][layer] >= env.base.cfg.packet_ttl_rounds)
        for node in np.flatnonzero(action)
        for layer in range(min(int(action[node]), len(env.base.packet_ages[node])))
    )


def _required_expiring_service(env, config: GuardedTeacherConfig) -> int:
    demand_after = int(env.step3_qos_counts["demand"]) + int(env.base.queue[env.members].sum())
    stale_before = int(env.step3_qos_counts["stale"])
    expiring_total = sum(
        int(age >= env.base.cfg.packet_ttl_rounds)
        for node in env.members for age in env.base.packet_ages[node]
    )
    allowed_stale_after = int(np.floor(config.stale_guard * max(1, demand_after)))
    return max(0, expiring_total - max(0, allowed_stale_after - stale_before))


def _repair_stale(env, action, caps, mask, costs, required) -> tuple[np.ndarray, list[dict]]:
    audit = []
    while _served_expiring(env, action) < required:
        receivers = [
            int(node) for node in np.flatnonzero(mask & (action < caps))
            if int(action[node]) < len(env.base.packet_ages[node])
            and env.base.packet_ages[node][int(action[node])] >= env.base.cfg.packet_ttl_rounds
        ]
        donors = [
            int(node) for node in np.flatnonzero(action > 0)
            if env.base.packet_ages[node][int(action[node]) - 1] < env.base.cfg.packet_ttl_rounds
        ]
        pairs = [(receiver, donor) for receiver in receivers for donor in donors if receiver != donor]
        if not pairs:
            break
        receiver, donor = min(
            pairs,
            key=lambda pair: (
                float(costs[pair[0]] - costs[pair[1]]),
                _position_key(env, pair[0]), _position_key(env, pair[1]),
            ),
        )
        action[donor] -= 1
        action[receiver] += 1
        audit.append({"donor": donor, "receiver": receiver, "cost_delta": float(costs[receiver] - costs[donor])})
    return action, audit


def _projected_fairness(env, action: np.ndarray) -> float:
    offered = env.step3_episode_offered_per_node.astype(np.float64) + env.base.queue
    delivered = env.step3_episode_delivered_per_node.astype(np.float64) + action
    return episode_service_fairness(delivered, offered)


def _repair_fairness(env, action, caps, mask, costs, target, required_expiring) -> tuple[np.ndarray, list[dict]]:
    """Apply only fairness-improving swaps, choosing least energy penalty on ties."""
    audit = []
    current = _projected_fairness(env, action)
    while current + 1e-15 < target:
        offered = env.step3_episode_offered_per_node.astype(np.float64) + env.base.queue
        delivered = env.step3_episode_delivered_per_node.astype(np.float64) + action
        eligible_fairness = offered > 0.0
        ratios = np.zeros_like(offered)
        ratios[eligible_fairness] = np.clip(
            delivered[eligible_fairness] / offered[eligible_fairness], 0.0, 1.0
        )
        ratio_sum = float(ratios[eligible_fairness].sum())
        ratio_square_sum = float(np.square(ratios[eligible_fairness]).sum())
        cohort = int(eligible_fairness.sum())
        served_expiring = _served_expiring(env, action)
        donors = [int(node) for node in np.flatnonzero(action > 0)]
        receivers = [int(node) for node in np.flatnonzero(mask & (action < caps))]
        best = None
        for donor in donors:
            for receiver in receivers:
                if donor == receiver:
                    continue
                removed_expiring = int(
                    env.base.packet_ages[donor][int(action[donor]) - 1]
                    >= env.base.cfg.packet_ttl_rounds
                )
                added_expiring = int(
                    env.base.packet_ages[receiver][int(action[receiver])]
                    >= env.base.cfg.packet_ttl_rounds
                )
                if served_expiring - removed_expiring + added_expiring < required_expiring:
                    continue
                donor_after = np.clip((delivered[donor] - 1.0) / offered[donor], 0.0, 1.0)
                receiver_after = np.clip((delivered[receiver] + 1.0) / offered[receiver], 0.0, 1.0)
                new_sum = ratio_sum - ratios[donor] - ratios[receiver] + donor_after + receiver_after
                new_square_sum = (
                    ratio_square_sum - ratios[donor] ** 2 - ratios[receiver] ** 2
                    + donor_after ** 2 + receiver_after ** 2
                )
                fairness = float(new_sum ** 2 / (cohort * new_square_sum)) if new_square_sum > 0.0 else 0.0
                gain = fairness - current
                if gain <= 1e-15:
                    continue
                key = (-gain, float(costs[receiver] - costs[donor]), _position_key(env, receiver), _position_key(env, donor))
                if best is None or key < best[0]:
                    best = (key, donor, receiver, fairness)
        if best is None:
            break
        _, donor, receiver, fairness = best
        action[donor] -= 1
        action[receiver] += 1
        audit.append({
            "donor": donor, "receiver": receiver, "fairness_before": current,
            "fairness_after": fairness, "cost_delta": float(costs[receiver] - costs[donor]),
        })
        current = fairness
    return action, audit


def guarded_marginal_action(
    env, mask, caps, budget: int, config: GuardedTeacherConfig, *, return_audit: bool = False,
):
    mask = np.asarray(mask, dtype=bool)
    caps = np.asarray(caps, dtype=np.int64)
    if mask.shape != caps.shape or len(caps) != env.base.n_nodes:
        raise ValueError("mask and caps must align with global node identity")
    target = min(int(budget), int(caps[mask].sum()))
    if target == 0:
        action = np.zeros_like(caps, dtype=np.int64)
        if return_audit:
            return action, {
                "target_slots": 0, "required_expiring_service": 0,
                "served_expiring": 0, "stale_guard_satisfied": True,
                "stale_swaps": [], "fairness_swaps": [],
                "projected_fairness": _projected_fairness(env, action),
                "fairness_guard_satisfied": True,
            }
        return action
    costs = _packet_costs(env, config)
    action = _base_allocation(env, mask, caps, target, costs)
    required = _required_expiring_service(env, config)
    action, stale_swaps = _repair_stale(env, action, caps, mask, costs, required)
    action, fairness_swaps = _repair_fairness(
        env, action, caps, mask, costs, config.fairness_guard, required
    )
    feasible = bool(
        action.shape == caps.shape and np.all(action >= 0) and np.all(action <= caps)
        and np.all(action[~mask] == 0) and int(action.sum()) == target
    )
    if not feasible:
        raise RuntimeError("teacher violated the exact feasible-service projection")
    if return_audit:
        projected_fairness = _projected_fairness(env, action)
        served_expiring = _served_expiring(env, action)
        return action, {
            "target_slots": target,
            "required_expiring_service": required,
            "served_expiring": served_expiring,
            "stale_guard_satisfied": served_expiring >= required,
            "stale_swaps": stale_swaps,
            "fairness_swaps": fairness_swaps,
            "projected_fairness": projected_fairness,
            "fairness_guard_satisfied": projected_fairness + 1e-15 >= config.fairness_guard,
        }
    return action
