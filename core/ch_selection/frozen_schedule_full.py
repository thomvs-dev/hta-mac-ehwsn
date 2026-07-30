"""Full-horizon frozen HEART-CH schedule generation with explicit coverage."""

from __future__ import annotations

import sys

import numpy as np

from .frozen_heart_ch import FrozenHeartCH
from ..reproducibility import set_global_seed


SCHEDULE_SCHEMA_VERSION = 2


def frozen_ch_schedule_full(
    policy: FrozenHeartCH,
    seed: int,
    *,
    horizon: int = 3000,
) -> dict:
    """Generate decisions up to ``horizon`` without silently repeating frames."""
    set_global_seed(seed)
    if str(policy.upstream) not in sys.path:
        sys.path.insert(0, str(policy.upstream))
    from train import load_stage1_params, make_env
    import config as cfg

    params = load_stage1_params(str(policy.upstream / cfg.STAGE1_PARAMS_PATH))
    env = make_env(params, seed=seed, mode=cfg.AGENT_MODE, max_rounds=horizon)
    state, _ = env.reset(seed=seed)
    edge_index, edge_weight = env.get_graph_data()
    alive_mask = env.get_alive_mask()
    frames: list[dict] = []
    stop_reason = "horizon_reached"
    terminal_alive_count = int(np.asarray(alive_mask, dtype=bool).sum())
    upstream_episode_stats: dict = {}
    upstream_termination_cause: str | None = None
    for _ in range(horizon):
        action, embedding = policy.select(
            state, edge_index, edge_weight, alive_mask
        )
        selected = np.flatnonzero(np.asarray(action) > 0).astype(np.int64)
        if selected.size == 0:
            terminal_alive_count = int(
                np.asarray(alive_mask, dtype=bool).sum()
            )
            stop_reason = (
                "frozen_policy_selected_no_ch"
                if terminal_alive_count > 0
                else "upstream_episode_terminated_before_selection"
            )
            if terminal_alive_count == 0:
                upstream_termination_cause = "all_nodes_dead"
            break
        frames.append(
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
        state, _, terminated, truncated, info = env.step(action)
        terminal_alive_count = int(info.get("alive_count", 0))
        upstream_episode_stats = dict(info.get("episode_stats", {}))
        if terminated:
            stop_reason = "upstream_episode_terminated"
            alive_ratio = terminal_alive_count / int(env.N)
            upstream_termination_cause = (
                "alive_fraction_below_death_threshold"
                if alive_ratio < float(env.death_threshold)
                else "no_surviving_current_ch"
            )
            break
        if truncated:
            stop_reason = "horizon_reached"
            break
        edge_index, edge_weight = env.get_graph_data()
        alive_mask = env.get_alive_mask()
    if not frames:
        raise RuntimeError("frozen HEART-CH generated an empty schedule")
    return {
        "frames": frames,
        "schedule_schema_version": SCHEDULE_SCHEMA_VERSION,
        "requested_horizon": int(horizon),
        "coverage_rounds": len(frames),
        "complete": len(frames) == horizon,
        "stop_reason": stop_reason,
        "terminal_alive_count": terminal_alive_count,
        "upstream_episode_stats": upstream_episode_stats,
        "upstream_termination_cause": upstream_termination_cause,
    }
