"""Greedy marginal-Q projection for the cluster slot-budget constraint."""

from __future__ import annotations

import heapq

import numpy as np


def project_slot_budget(
    branch_q_values,
    budget: int,
    *,
    stop_at_nonpositive_gain: bool = True,
) -> np.ndarray:
    """Project branch Q values onto sum(a_i) <= budget."""
    q = np.asarray(branch_q_values, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < 1:
        raise ValueError("branch_q_values must have shape [nodes, n_max+1]")
    if budget < 0 or not np.all(np.isfinite(q)):
        raise ValueError("budget must be non-negative and Q values finite")

    allocation = np.zeros(q.shape[0], dtype=np.int64)
    if q.shape[1] == 1 or budget == 0:
        return allocation

    heap: list[tuple[float, int, int]] = []
    for node in range(q.shape[0]):
        gain = q[node, 1] - q[node, 0]
        heapq.heappush(heap, (-float(gain), node, 1))

    for _ in range(int(budget)):
        if not heap:
            break
        neg_gain, node, level = heapq.heappop(heap)
        gain = -neg_gain
        if stop_at_nonpositive_gain and gain <= 0.0:
            break
        allocation[node] = level
        next_level = level + 1
        if next_level < q.shape[1]:
            next_gain = q[node, next_level] - q[node, level]
            heapq.heappush(heap, (-float(next_gain), node, next_level))

    return allocation
