"""Generate an immutable per-round CH schedule from frozen HEART-CH."""

from __future__ import annotations

import sys

import numpy as np

from .frozen_heart_ch import FrozenHeartCH


def frozen_ch_schedule(
    policy: FrozenHeartCH,
    seed: int,
    *,
    max_rounds: int = 2000,
) -> list[dict]:
    """Run upstream HEART-CH without retraining and archive every CH decision."""
    if str(policy.upstream) not in sys.path:
        sys.path.insert(0, str(policy.upstream))
    from train import load_stage1_params, make_env
    import config as cfg

    params = load_stage1_params(str(policy.upstream / cfg.STAGE1_PARAMS_PATH))
    env = make_env(
        params, seed=seed, mode=cfg.AGENT_MODE, max_rounds=max_rounds
    )
    state, _ = env.reset(seed=seed)
    edge_index, edge_weight = env.get_graph_data()
    alive_mask = env.get_alive_mask()
    schedule: list[dict] = []
    for _ in range(max_rounds):
        action, embedding = policy.select(
            state, edge_index, edge_weight, alive_mask
        )
        selected = np.flatnonzero(np.asarray(action) > 0).astype(np.int64)
        if selected.size == 0:
            break
        schedule.append(
            {
                "positions": np.asarray(
                    env.node_positions, dtype=np.float64
                ).copy(),
                "solar_states": np.asarray(
                    env.hmm_states, dtype=np.int64
                ).copy(),
                "thermal_states": np.asarray(
                    env.thermal_hmm_states, dtype=np.int64
                ).copy(),
                "cluster_heads": selected,
                "stgcn_embedding": np.asarray(embedding).copy(),
            }
        )
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
        edge_index, edge_weight = env.get_graph_data()
        alive_mask = env.get_alive_mask()
    if not schedule:
        raise RuntimeError("frozen HEART-CH generated an empty schedule")
    return schedule
