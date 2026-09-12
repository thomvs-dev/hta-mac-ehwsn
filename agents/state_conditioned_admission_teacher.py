"""Observable energy-risk admission shield for the deterministic HTA-MAC teacher.

The shield starts from the full-service idle-aware action.  It contracts the
current target-cluster frame only when the no-harvest post-action energy of a
transmitting member or the current CH falls below a configured reserve.  Any
contracted action must preserve the full action's expiring-packet service and
must not reduce projected episode-service Jain fairness.  HEART-CH remains
fixed and exogenous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents.idle_aware_oracle_teacher import (
    IdleAwareOracleTeacherConfig,
    idle_aware_oracle_action,
)
from core.energy.idle_model import idle_listening_energy


@dataclass(frozen=True)
class StateConditionedAdmissionConfig:
    allocator: IdleAwareOracleTeacherConfig
    reserve_fraction: float
    maximum_slot_reduction: int
    minimum_service_cap: int = 20

    @classmethod
    def from_payload(cls, payload: dict) -> "StateConditionedAdmissionConfig":
        reserve = float(payload["reserve_fraction"])
        reduction = int(payload["maximum_slot_reduction"])
        minimum = int(payload.get("minimum_service_cap", 20))
        if not 0.0 <= reserve <= 1.0:
            raise ValueError("reserve_fraction must lie in [0, 1]")
        if not 0 <= reduction <= 4:
            raise ValueError("maximum_slot_reduction must lie in [0, 4]")
        if minimum < 1:
            raise ValueError("minimum_service_cap must be positive")
        return cls(
            allocator=IdleAwareOracleTeacherConfig.from_payload(payload["allocator"]),
            reserve_fraction=reserve,
            maximum_slot_reduction=reduction,
            minimum_service_cap=minimum,
        )


def _no_harvest_energy_profile(env, action: np.ndarray) -> dict:
    action = np.asarray(action, dtype=np.int64)
    active = np.flatnonzero(action > 0)
    frame_slots = int(action.sum())
    idle_per_slot = float(idle_listening_energy(
        1,
        p_idle_j_per_bit_time=env.base.cfg.e_elec_j_per_bit,
        slot_bit_times=env.base.cfg.idle_slot_bit_times,
    ))
    post_values = []
    member_consumption = 0.0
    for node in active:
        distance = float(np.linalg.norm(env.base.positions[node] - env.base.positions[int(env.ch)]))
        tx = float(env.base.radio.tx(env.base.cfg.packet_bits * int(action[node]), distance))
        idle = float(max(0, frame_slots - int(action[node])) * idle_per_slot)
        consumption = tx + idle
        member_consumption += consumption
        post_values.append(float(env.base.energy[node]) - consumption)
    ch_consumption = 0.0
    if frame_slots > 0 and env.base.alive[int(env.ch)]:
        bits = env.base.cfg.packet_bits * frame_slots
        ch_consumption = float(env.base.radio.rx(bits) + env.base.radio.aggregate(bits))
        distance_bs = float(np.linalg.norm(
            env.base.positions[int(env.ch)] - np.asarray(env.base.cfg.bs_position_m)
        ))
        ch_consumption += float(env.base.radio.tx(env.base.cfg.packet_bits, distance_bs))
        post_values.append(float(env.base.energy[int(env.ch)]) - ch_consumption)
    minimum_post = min(post_values) if post_values else float("inf")
    initial = max(float(env.base.cfg.initial_energy_j), 1e-12)
    return {
        "minimum_controlled_post_energy_j": float(minimum_post),
        "minimum_controlled_post_energy_fraction": float(minimum_post / initial),
        "member_consumption_j": float(member_consumption),
        "ch_consumption_j": float(ch_consumption),
        "total_controlled_consumption_j": float(member_consumption + ch_consumption),
    }


def state_conditioned_admission_action(
    env,
    mask,
    caps,
    budget: int,
    config: StateConditionedAdmissionConfig,
    *,
    return_audit: bool = False,
):
    """Apply observable energy-risk frame contraction to the full-service action."""
    full_action, full_audit = idle_aware_oracle_action(
        env, mask, caps, budget, config.allocator, return_audit=True
    )
    full_profile = _no_harvest_energy_profile(env, full_action)
    full_base = full_audit["base_teacher"]
    floor = config.reserve_fraction
    selected_action = full_action
    selected_audit = full_audit
    selected_profile = full_profile
    selected_cap = int(budget)
    candidates_considered = 1
    constrained_candidates = 1

    if (
        int(full_action.sum()) > 0
        and full_profile["minimum_controlled_post_energy_fraction"] < floor
        and config.maximum_slot_reduction > 0
    ):
        lower = max(config.minimum_service_cap, int(budget) - config.maximum_slot_reduction)
        alternatives = []
        for service_cap in range(int(budget) - 1, lower - 1, -1):
            action, audit = idle_aware_oracle_action(
                env, mask, caps, service_cap, config.allocator, return_audit=True
            )
            candidates_considered += 1
            base = audit["base_teacher"]
            constraint_ok = bool(
                base["served_expiring"] >= full_base["served_expiring"]
                and base["projected_fairness"] + 1e-12 >= full_base["projected_fairness"]
            )
            if not constraint_ok:
                continue
            constrained_candidates += 1
            profile = _no_harvest_energy_profile(env, action)
            alternatives.append((service_cap, action, audit, profile))
        meeting_floor = [row for row in alternatives if row[3]["minimum_controlled_post_energy_fraction"] >= floor]
        eligible = meeting_floor if meeting_floor else alternatives
        if eligible:
            # Preserve as much immediate service as possible once the reserve is
            # met; otherwise maximize the observable safety margin.
            if meeting_floor:
                selected_cap, selected_action, selected_audit, selected_profile = max(
                    eligible, key=lambda row: row[0]
                )
            else:
                selected_cap, selected_action, selected_audit, selected_profile = max(
                    eligible,
                    key=lambda row: (row[3]["minimum_controlled_post_energy_fraction"], row[0]),
                )

    audit = {
        "triggered": bool(selected_cap < int(budget)),
        "selected_service_cap": int(selected_cap),
        "allocated_slots": int(selected_action.sum()),
        "full_allocated_slots": int(full_action.sum()),
        "candidates_considered": int(candidates_considered),
        "constraint_feasible_candidates": int(constrained_candidates),
        "full_profile": full_profile,
        "selected_profile": selected_profile,
        "full_served_expiring": int(full_base["served_expiring"]),
        "selected_served_expiring": int(selected_audit["base_teacher"]["served_expiring"]),
        "full_projected_fairness": float(full_base["projected_fairness"]),
        "selected_projected_fairness": float(selected_audit["base_teacher"]["projected_fairness"]),
        "allocator": selected_audit,
    }
    return (selected_action, audit) if return_audit else selected_action
