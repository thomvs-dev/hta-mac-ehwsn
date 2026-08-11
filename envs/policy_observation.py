"""Versioned policy observations for HTA-MAC training wrappers."""

from __future__ import annotations

import numpy as np


LEGACY_POLICY_SCHEMA = "phase2c_v1"
PHASE2D_POLICY_SCHEMA = "phase2d_ttl_cap_v2"
SUPPORTED_POLICY_SCHEMAS = (LEGACY_POLICY_SCHEMA, PHASE2D_POLICY_SCHEMA)
BASE_PHYSICAL_FEATURES = 18


def packet_age_histogram(base_env) -> np.ndarray:
    """Return normalized per-node FIFO age counts for ages 0 through TTL."""
    ttl = int(base_env.cfg.packet_ttl_rounds)
    queue_max = int(base_env.cfg.queue_max_packets)
    if ttl < 0 or queue_max <= 0:
        raise ValueError("invalid queue TTL/capacity for policy observation")
    result = np.zeros((base_env.n_nodes, ttl + 1), dtype=np.float32)
    for node, ages in enumerate(base_env.packet_ages):
        age_array = np.asarray(ages, dtype=np.int64)
        if age_array.size and (
            np.any(age_array < 0) or np.any(age_array > ttl)
        ):
            raise RuntimeError("packet age outside the retained TTL range")
        if age_array.size:
            result[node] = np.bincount(
                age_array, minlength=ttl + 1
            )[: ttl + 1]
    result /= float(queue_max)
    return result


def action_validity_features(base_env, active_mask) -> np.ndarray:
    """Encode queue-feasible actions without exposing a learned node ID."""
    active = np.asarray(active_mask, dtype=bool)
    if active.shape != (base_env.n_nodes,):
        raise ValueError("active mask must preserve global node identity")
    actions = int(base_env.cfg.n_max) + 1
    caps = np.minimum(base_env.queue, base_env.cfg.n_max).astype(np.int64)
    caps[~active] = 0
    levels = np.arange(actions, dtype=np.int64)[None, :]
    validity = levels <= caps[:, None]
    validity &= active[:, None]
    return validity.astype(np.float32)


def policy_feature_layout(base_env, schema: str) -> dict[str, object]:
    """Describe a schema without depending on a particular observation."""
    if schema not in SUPPORTED_POLICY_SCHEMAS:
        raise ValueError(f"unsupported policy observation schema: {schema}")
    embedding_dim = int(np.asarray(base_env.embedding).shape[1])
    if schema == LEGACY_POLICY_SCHEMA:
        return {
            "schema": schema,
            "physical_features": BASE_PHYSICAL_FEATURES,
            "packet_age_features": 0,
            "action_validity_features": 0,
            "embedding_features": embedding_dim,
            "embedding_start": BASE_PHYSICAL_FEATURES,
            "total_features": BASE_PHYSICAL_FEATURES + embedding_dim,
        }
    age_features = int(base_env.cfg.packet_ttl_rounds) + 1
    action_features = int(base_env.cfg.n_max) + 1
    embedding_start = BASE_PHYSICAL_FEATURES + age_features + action_features
    return {
        "schema": schema,
        "physical_features": BASE_PHYSICAL_FEATURES,
        "packet_age_features": age_features,
        "action_validity_features": action_features,
        "embedding_features": embedding_dim,
        "embedding_start": embedding_start,
        "total_features": embedding_start + embedding_dim,
    }


def build_policy_observation(
    base_env,
    physical_state,
    active_mask,
    *,
    schema: str,
    node_indices=None,
) -> np.ndarray:
    """Build a branch-aligned legacy or Phase 2D observation tensor."""
    physical = np.asarray(physical_state, dtype=np.float32)
    if physical.shape != (base_env.n_nodes, BASE_PHYSICAL_FEATURES):
        raise ValueError("physical state does not match the 18-feature schema")
    active = np.asarray(active_mask, dtype=bool)
    if active.shape != (base_env.n_nodes,):
        raise ValueError("active mask must preserve global node identity")
    nodes = (
        np.arange(base_env.n_nodes, dtype=np.int64)
        if node_indices is None
        else np.asarray(node_indices, dtype=np.int64)
    )
    embedding = np.asarray(base_env.embedding, dtype=np.float32)
    blocks = [physical[nodes]]
    if schema == PHASE2D_POLICY_SCHEMA:
        blocks.extend(
            (
                packet_age_histogram(base_env)[nodes],
                action_validity_features(base_env, active)[nodes],
            )
        )
    elif schema != LEGACY_POLICY_SCHEMA:
        raise ValueError(f"unsupported policy observation schema: {schema}")
    blocks.append(embedding[nodes])
    observation = np.concatenate(blocks, axis=1).astype(np.float32)
    expected = int(policy_feature_layout(base_env, schema)["total_features"])
    if observation.shape != (len(nodes), expected):
        raise RuntimeError("policy observation schema produced an invalid shape")
    if not np.all(np.isfinite(observation)):
        raise RuntimeError("policy observation contains non-finite values")
    return observation
