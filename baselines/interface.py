"""Shared policy contract for all Phase 3 MAC comparisons."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MACPolicyInterface(ABC):
    """A stateless-or-resettable allocator operating inside one MAC env."""

    name: str
    literature_baseline: bool = True
    comparison_role: str = "literature_baseline"

    def reset(self, seed: int) -> None:
        """Reset policy-local randomness or history for one paired trial."""

    @abstractmethod
    def select_action(self, state: np.ndarray, env) -> np.ndarray:
        """Return integer per-node slots satisfying the environment budget."""

    @staticmethod
    def eligible_members(env, cluster: int, ch: int) -> np.ndarray:
        if not env.alive[ch]:
            return np.empty(0, dtype=np.int64)
        node_ids = np.arange(env.n_nodes)
        return np.flatnonzero(
            (env.cluster_of == cluster) & env.alive & (node_ids != ch)
        )

    @staticmethod
    def validate(action: np.ndarray, env) -> np.ndarray:
        action = np.asarray(action, dtype=np.int64)
        env._validate_action(action)
        return action
