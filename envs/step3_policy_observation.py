"""Identity-safe Step 3 v3 observations with explicit scheduled-CH context."""

from __future__ import annotations

import numpy as np

from envs.policy_observation import PHASE2D_POLICY_SCHEMA, build_policy_observation, policy_feature_layout


STEP3_CH_CONTEXT_SCHEMA = "step3_ch_context_v3"
STEP3_WORKLOAD_CONTEXT_SCHEMA = "step3_workload_context_v4"
CH_CONTEXT_NAMES = (
    "scheduled_ch_reserve_fraction",
    "scheduled_ch_forecast_fraction",
    "scheduled_ch_forecast_uncertainty_fraction",
    "scheduled_ch_bs_distance_fraction",
    "target_feasible_backlog_fraction",
    "target_member_fraction",
    "scheduled_ch_alive",
)
WORKLOAD_CONTEXT_NAMES = (
    "target_backlog_per_member_fraction",
    "target_budget_pressure_log_fraction",
    "target_expiring_packet_fraction",
    "target_demand_ewma_fraction",
    "network_alive_fraction",
    "target_member_energy_p10_fraction",
    "qos_lifetime_preference",
)


def scheduled_ch_context(base_env, ch: int, members, physical_state, risk_config) -> np.ndarray:
    ch = int(ch)
    members = np.asarray(members, dtype=np.int64)
    state = np.asarray(physical_state, dtype=np.float64)
    forecast_reference = max(float(risk_config["forecast_reference_j"]), 1e-12)
    distance_reference = max(float(risk_config["distance_reference_m"]), 1e-12)
    feasible = int(np.minimum(base_env.queue[members], base_env.cfg.n_max).sum())
    context = np.asarray(
        [
            np.clip(base_env.energy[ch] / base_env.cfg.initial_energy_j, 0.0, 1.0),
            np.clip(state[ch, 1] / forecast_reference, 0.0, 1.0),
            np.clip(np.sqrt(max(0.0, state[ch, 2])) / forecast_reference, 0.0, 1.0),
            np.clip(
                np.linalg.norm(base_env.positions[ch] - np.asarray(base_env.cfg.bs_position_m))
                / distance_reference,
                0.0,
                1.0,
            ),
            np.clip(feasible / max(1, base_env.cfg.frame_slot_budget), 0.0, 1.0),
            np.clip(len(members) / max(1, base_env.n_nodes - 1), 0.0, 1.0),
            float(base_env.alive[ch]),
        ],
        dtype=np.float32,
    )
    if context.shape != (len(CH_CONTEXT_NAMES),) or not np.all(np.isfinite(context)):
        raise RuntimeError("invalid scheduled-CH context")
    return context


def workload_context(base_env, members, wrapper=None) -> np.ndarray:
    members = np.asarray(members, dtype=np.int64)
    alive_members = members[base_env.alive[members]]
    member_count = max(1, len(alive_members))
    backlog = float(base_env.queue[alive_members].sum())
    queue_capacity = max(1.0, member_count * float(base_env.cfg.queue_max_packets))
    budget = max(1.0, float(base_env.cfg.frame_slot_budget))
    expiring = float(sum(
        sum(age >= base_env.cfg.packet_ttl_rounds for age in base_env.packet_ages[node])
        for node in alive_members
    ))
    demand_ewma = float(getattr(wrapper, "step3_demand_ewma", backlog / member_count))
    preference = float(getattr(wrapper, "step3_qos_lifetime_preference", 0.5))
    energy = (
        base_env.energy[alive_members] / max(float(base_env.cfg.initial_energy_j), 1e-12)
        if len(alive_members) else np.asarray([0.0])
    )
    context = np.asarray([
        np.clip(backlog / queue_capacity, 0.0, 1.0),
        np.clip(np.log1p(backlog / budget) / np.log(5.0), 0.0, 1.0),
        np.clip(expiring / queue_capacity, 0.0, 1.0),
        np.clip(demand_ewma / max(1.0, float(base_env.cfg.queue_max_packets)), 0.0, 1.0),
        np.clip(np.mean(base_env.alive), 0.0, 1.0),
        np.clip(np.quantile(energy, 0.10), 0.0, 1.0),
        np.clip(preference, 0.0, 1.0),
    ], dtype=np.float32)
    if context.shape != (len(WORKLOAD_CONTEXT_NAMES),) or not np.all(np.isfinite(context)):
        raise RuntimeError("invalid workload context")
    return context


def step3_observation_layout(base_env, schema=STEP3_CH_CONTEXT_SCHEMA) -> dict:
    if schema not in {STEP3_CH_CONTEXT_SCHEMA, STEP3_WORKLOAD_CONTEXT_SCHEMA}:
        raise ValueError(f"unsupported Step 3 observation schema: {schema}")
    base = policy_feature_layout(base_env, PHASE2D_POLICY_SCHEMA)
    names = list(CH_CONTEXT_NAMES)
    if schema == STEP3_WORKLOAD_CONTEXT_SCHEMA:
        names.extend(WORKLOAD_CONTEXT_NAMES)
    embedding_start = int(base["embedding_start"]) + len(names)
    return {
        "schema": schema,
        "physical_features": base["physical_features"],
        "packet_age_features": base["packet_age_features"],
        "action_validity_features": base["action_validity_features"],
        "scheduled_ch_context_features": len(names),
        "scheduled_ch_context_names": names,
        "embedding_features": base["embedding_features"],
        "embedding_start": embedding_start,
        "total_features": int(base["total_features"]) + len(names),
    }


def build_step3_observation(
    base_env, physical_state, active_mask, *, ch, members, risk_config,
    schema=STEP3_CH_CONTEXT_SCHEMA, wrapper=None,
):
    base = build_policy_observation(
        base_env, physical_state, active_mask, schema=PHASE2D_POLICY_SCHEMA
    )
    old_layout = policy_feature_layout(base_env, PHASE2D_POLICY_SCHEMA)
    split = int(old_layout["embedding_start"])
    context = scheduled_ch_context(base_env, ch, members, physical_state, risk_config)
    if schema == STEP3_WORKLOAD_CONTEXT_SCHEMA:
        context = np.concatenate((context, workload_context(base_env, members, wrapper)))
    elif schema != STEP3_CH_CONTEXT_SCHEMA:
        raise ValueError(f"unsupported Step 3 observation schema: {schema}")
    broadcast = np.broadcast_to(context, (base_env.n_nodes, context.size)).copy()
    observation = np.concatenate((base[:, :split], broadcast, base[:, split:]), axis=1).astype(np.float32)
    expected = step3_observation_layout(base_env, schema)
    if observation.shape != (base_env.n_nodes, expected["total_features"]):
        raise RuntimeError("Step 3 observation shape mismatch")
    return observation
