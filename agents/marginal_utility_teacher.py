"""Deterministic marginal service/energy teacher for HTA-MAC V2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarginalTeacherConfig:
    energy_weight: float
    stale_weight: float
    fairness_weight: float
    age_weight: float = 0.25
    queue_weight: float = 0.10
    residual_weight: float = 0.20
    harvest_weight: float = 0.10
    diminishing_weight: float = 0.15
    member_risk_weight: float = 0.50
    ch_risk_weight: float = 0.50
    member_reserve_floor: float = 0.05
    ch_reserve_floor: float = 0.05

    @classmethod
    def from_payload(cls, payload: dict) -> "MarginalTeacherConfig":
        result = cls(**{name: float(payload[name]) for name in cls.__dataclass_fields__})
        values = np.asarray(list(result.__dict__.values()), dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("teacher coefficients must be finite and nonnegative")
        if result.member_reserve_floor > 1.0 or result.ch_reserve_floor > 1.0:
            raise ValueError("reserve floors must lie in [0,1]")
        return result


def _risk(post_fraction: float, floor: float) -> float:
    if floor <= 0.0:
        return 0.0
    return float(max(0.0, floor - post_fraction) / floor)


def marginal_candidates(env, mask, caps, action, config: MarginalTeacherConfig) -> list[dict]:
    """Return finite permutation-safe features for every feasible next slot."""
    base = env.base
    mask = np.asarray(mask, dtype=bool)
    caps = np.asarray(caps, dtype=np.int64)
    action = np.asarray(action, dtype=np.int64)
    if mask.shape != caps.shape or caps.shape != action.shape or len(action) != base.n_nodes:
        raise ValueError("mask, caps, and action must align with global node identity")
    eligible = np.flatnonzero(mask & (action < caps))
    if not len(eligible):
        return []
    ch = int(env.ch)
    initial = max(float(base.cfg.initial_energy_j), 1e-12)
    offered = env.step3_episode_offered_per_node.astype(np.float64)
    delivered = env.step3_episode_delivered_per_node.astype(np.float64)
    service_ratio = np.divide(delivered, np.maximum(offered, 1.0))
    active_ratio = service_ratio[mask]
    maximum_ratio = float(active_ratio.max(initial=0.0))
    forecast = np.maximum(base._state()[:, 1].astype(np.float64), 0.0)
    forecast_scale = max(float(forecast[mask].max(initial=0.0)), 1e-12)
    distance_bs = float(np.linalg.norm(base.positions[ch] - np.asarray(base.cfg.bs_position_m)))
    forwarding_setup = float(base.radio.tx(base.cfg.packet_bits, distance_bs))
    per_packet_ch = float(
        base.radio.rx(base.cfg.packet_bits) + base.radio.aggregate(base.cfg.packet_bits)
    )
    candidates = []
    for node in eligible:
        layer = int(action[node]) + 1
        if layer > len(base.packet_ages[node]):
            raise RuntimeError("queue cap exceeds concrete packet-age queue")
        age = int(base.packet_ages[node][layer - 1])
        distance = float(np.linalg.norm(base.positions[node] - base.positions[ch]))
        member_tx = float(base.radio.tx(base.cfg.packet_bits, distance))
        ch_setup = forwarding_setup if int(action.sum()) == 0 else 0.0
        marginal_energy = member_tx + per_packet_ch + ch_setup
        member_post = (float(base.energy[node]) - member_tx) / initial
        served_after = int(action.sum()) + 1
        ch_cost_after = per_packet_ch * served_after + forwarding_setup
        ch_post = (float(base.energy[ch]) - ch_cost_after) / initial
        fairness_deficit = max(0.0, maximum_ratio - float(service_ratio[node]))
        position = tuple(float(value) for value in base.positions[node])
        candidates.append({
            "node": int(node), "layer": layer, "packet_age": age,
            "expiring": float(age >= base.cfg.packet_ttl_rounds),
            "age_fraction": float(age / max(1, base.cfg.packet_ttl_rounds)),
            "fairness_deficit": float(fairness_deficit),
            "queue_fraction": float((int(caps[node]) - int(action[node])) / max(1, base.cfg.queue_max_packets)),
            "residual_fraction": float(np.clip(base.energy[node] / initial, 0.0, 1.0)),
            "harvest_fraction": float(forecast[node] / forecast_scale),
            "member_risk": _risk(member_post, config.member_reserve_floor),
            "ch_risk": _risk(ch_post, config.ch_reserve_floor),
            "marginal_energy_j": marginal_energy,
            "member_tx_energy_j": member_tx,
            "ch_packet_energy_j": per_packet_ch,
            "ch_forwarding_setup_j": ch_setup,
            "position": position,
        })
    energy_scale = max(float(np.median([row["marginal_energy_j"] for row in candidates])), 1e-12)
    for row in candidates:
        row["normalized_energy"] = float(row["marginal_energy_j"] / energy_scale)
        row["score"] = float(
            config.stale_weight * row["expiring"]
            + config.fairness_weight * row["fairness_deficit"]
            + config.age_weight * row["age_fraction"]
            + config.queue_weight * row["queue_fraction"]
            + config.residual_weight * row["residual_fraction"]
            + config.harvest_weight * row["harvest_fraction"]
            - config.energy_weight * row["normalized_energy"]
            - config.member_risk_weight * row["member_risk"]
            - config.ch_risk_weight * row["ch_risk"]
            - config.diminishing_weight * float(action[row["node"]])
        )
        if not np.isfinite(row["score"]):
            raise RuntimeError("teacher produced a non-finite marginal score")
    return candidates


def _candidate_key(row: dict) -> tuple:
    # Physical/local fields travel with a node under permutation. Node index is
    # intentionally absent; exact physical duplicates are allocation-symmetric.
    return (
        row["score"], row["expiring"], row["fairness_deficit"],
        -row["normalized_energy"], row["residual_fraction"],
        row["harvest_fraction"], -row["position"][0], -row["position"][1],
    )


def marginal_utility_action(
    env, mask, caps, budget: int, config: MarginalTeacherConfig,
    *, return_audit: bool = False,
):
    """Greedily fill feasible service by the declared marginal score."""
    mask = np.asarray(mask, dtype=bool)
    caps = np.asarray(caps, dtype=np.int64)
    action = np.zeros_like(caps)
    decisions = []
    target = min(int(budget), int(caps[mask].sum()))
    for _ in range(target):
        candidates = marginal_candidates(env, mask, caps, action, config)
        if not candidates:
            break
        winner = max(candidates, key=_candidate_key)
        action[int(winner["node"])] += 1
        if return_audit:
            decisions.append({key: value for key, value in winner.items() if key != "position"})
    feasible = bool(
        action.shape == caps.shape
        and np.all(action >= 0) and np.all(action <= caps)
        and np.all(action[~mask] == 0) and int(action.sum()) == target
    )
    if not feasible:
        raise RuntimeError("teacher violated the exact feasible-service projection")
    return (action, decisions) if return_audit else action
