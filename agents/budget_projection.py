"""Greedy marginal-Q projection for the cluster slot-budget constraint."""

from __future__ import annotations

import heapq

import numpy as np


def project_slot_budget(
    branch_q_values,
    budget: int,
    *,
    stop_at_nonpositive_gain: bool = True,
    tie_break_priorities=None,
) -> np.ndarray:
    """Project branch Q values onto sum(a_i) <= budget."""
    q = np.asarray(branch_q_values, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < 1:
        raise ValueError("branch_q_values must have shape [nodes, n_max+1]")
    if budget < 0 or not np.all(np.isfinite(q)):
        raise ValueError("budget must be non-negative and Q values finite")

    if tie_break_priorities is None:
        priorities = np.arange(q.shape[0], dtype=np.int64)
    else:
        priorities = np.asarray(tie_break_priorities, dtype=np.int64)
        if priorities.shape != (q.shape[0],):
            raise ValueError("tie-break priorities must align with branches")
        if np.unique(priorities).size != priorities.size:
            raise ValueError("tie-break priorities must be unique")

    allocation = np.zeros(q.shape[0], dtype=np.int64)
    if q.shape[1] == 1 or budget == 0:
        return allocation

    heap: list[tuple[float, int, int, int]] = []
    for node in range(q.shape[0]):
        gain = q[node, 1] - q[node, 0]
        heapq.heappush(
            heap, (-float(gain), int(priorities[node]), node, 1)
        )

    for _ in range(int(budget)):
        if not heap:
            break
        neg_gain, _, node, level = heapq.heappop(heap)
        gain = -neg_gain
        if stop_at_nonpositive_gain and gain <= 0.0:
            break
        allocation[node] = level
        next_level = level + 1
        if next_level < q.shape[1]:
            next_gain = q[node, next_level] - q[node, level]
            heapq.heappush(
                heap,
                (-float(next_gain), int(priorities[node]), node, next_level),
            )

    return allocation


def solve_slot_budget_exact(branch_q_values, budget: int, *, caps=None) -> tuple[np.ndarray, float]:
    """Solve the separable integer slot allocation exactly by dynamic programming.

    Unlike :func:`project_slot_budget`, this routine does not assume diminishing
    marginal values.  It is intended for diagnostic comparison, not the online
    control path.  The returned objective is ``sum_i Q_i(a_i)``.
    """
    q = np.asarray(branch_q_values, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] < 1:
        raise ValueError("branch_q_values must have shape [nodes, n_max+1]")
    if budget < 0 or not np.all(np.isfinite(q)):
        raise ValueError("budget must be non-negative and Q values finite")
    nodes, actions = q.shape
    if caps is None:
        caps_array = np.full(nodes, actions - 1, dtype=np.int64)
    else:
        caps_array = np.asarray(caps, dtype=np.int64)
        if caps_array.shape != (nodes,):
            raise ValueError("caps must align with branches")
        if np.any(caps_array < 0) or np.any(caps_array >= actions):
            raise ValueError("caps outside the available action range")

    effective_budget = min(int(budget), int(caps_array.sum()))
    values = np.full((nodes + 1, effective_budget + 1), -np.inf, dtype=np.float64)
    choices = np.zeros((nodes, effective_budget + 1), dtype=np.int64)
    values[0, 0] = 0.0
    for node in range(nodes):
        for used in range(effective_budget + 1):
            best_value = -np.inf
            best_action = 0
            for action in range(min(int(caps_array[node]), used) + 1):
                previous = values[node, used - action]
                candidate = previous + q[node, action]
                if candidate > best_value:
                    best_value = candidate
                    best_action = action
            values[node + 1, used] = best_value
            choices[node, used] = best_action

    used = int(np.argmax(values[nodes]))
    allocation = np.zeros(nodes, dtype=np.int64)
    for node in range(nodes - 1, -1, -1):
        action = int(choices[node, used])
        allocation[node] = action
        used -= action
    objective = float(sum(q[node, allocation[node]] for node in range(nodes)))
    return allocation, objective


def projection_optimality_diagnostic(branch_q_values, allocation, budget: int, *, caps=None) -> dict:
    """Compare a feasible allocation with the exact separable-Q optimum."""
    q = np.asarray(branch_q_values, dtype=np.float64)
    candidate = np.asarray(allocation, dtype=np.int64)
    if candidate.shape != (q.shape[0],):
        raise ValueError("allocation must align with branches")
    caps_array = (
        np.full(q.shape[0], q.shape[1] - 1, dtype=np.int64)
        if caps is None else np.asarray(caps, dtype=np.int64)
    )
    feasible = bool(
        caps_array.shape == candidate.shape
        and np.all(candidate >= 0)
        and np.all(candidate <= caps_array)
        and int(candidate.sum()) <= int(budget)
    )
    if not feasible:
        raise ValueError("candidate allocation is infeasible")
    exact, exact_value = solve_slot_budget_exact(q, budget, caps=caps_array)
    candidate_value = float(sum(q[node, candidate[node]] for node in range(q.shape[0])))
    regret = max(0.0, exact_value - candidate_value)
    return {
        "candidate_objective": candidate_value,
        "exact_objective": exact_value,
        "absolute_regret": float(regret),
        "allocation_match": bool(np.array_equal(candidate, exact)),
        "candidate_slots": int(candidate.sum()),
        "exact_slots": int(exact.sum()),
    }
