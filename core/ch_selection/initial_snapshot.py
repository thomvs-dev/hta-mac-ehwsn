"""One-shot frozen HEART-CH cluster snapshot adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from .frozen_heart_ch import FrozenHeartCH


def frozen_initial_snapshot(policy: FrozenHeartCH, seed: int) -> dict:
    """Run the immutable upstream CH policy once at episode reset."""
    if str(policy.upstream) not in sys.path:
        sys.path.insert(0, str(policy.upstream))
    from train import load_stage1_params, make_env
    import config as cfg

    params = load_stage1_params(str(policy.upstream / cfg.STAGE1_PARAMS_PATH))
    env = make_env(params, seed=seed, mode=cfg.AGENT_MODE, max_rounds=1)
    state, _ = env.reset(seed=seed)
    edge_index, edge_weight = env.get_graph_data()
    action, embedding = policy.select(
        state, edge_index, edge_weight, env.get_alive_mask()
    )
    selected = np.flatnonzero(np.asarray(action) > 0)
    if selected.size == 0:
        raise RuntimeError("frozen HEART-CH selected no cluster heads")
    return {
        "positions": np.asarray(env.node_positions, dtype=np.float64).copy(),
        "solar_states": np.asarray(env.hmm_states, dtype=np.int64).copy(),
        "thermal_states": np.asarray(env.thermal_hmm_states, dtype=np.int64).copy(),
        "cluster_heads": selected.astype(np.int64),
        "stgcn_embedding": np.asarray(embedding).copy(),
    }
