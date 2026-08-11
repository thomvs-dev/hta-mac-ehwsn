"""Permutation-safe branch bundles for identity-robustness audits.

The permutation convention is ``permuted[i] = original[permutation[i]]``.
Consequently, branch-indexed outputs are returned to physical-node order with
``physical[permutation[i]] = permuted_output[i]``.
"""

from __future__ import annotations

import numpy as np


def action_mask_from_caps(mask, caps, actions: int) -> np.ndarray:
    """Build the complete per-branch discrete-action validity mask."""
    validity = np.asarray(mask, dtype=bool)
    limits = np.asarray(caps, dtype=np.int64)
    if validity.ndim != 1 or limits.shape != validity.shape:
        raise ValueError("mask and caps must be aligned one-dimensional arrays")
    if actions <= 0:
        raise ValueError("actions must be positive")
    levels = np.arange(int(actions), dtype=np.int64)[None, :]
    result = levels <= np.clip(limits, 0, int(actions) - 1)[:, None]
    result &= validity[:, None]
    return result


def validate_permutation(permutation, size: int) -> np.ndarray:
    order = np.asarray(permutation, dtype=np.int64)
    if order.shape != (int(size),):
        raise ValueError("permutation shape mismatch")
    if not np.array_equal(np.sort(order), np.arange(size, dtype=np.int64)):
        raise ValueError("indices do not form a complete permutation")
    return order


def active_branch_permutation(mask, rng: np.random.Generator) -> np.ndarray:
    """Permute active branch positions and leave inactive padding fixed."""
    validity = np.asarray(mask, dtype=bool)
    if validity.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    order = np.arange(validity.size, dtype=np.int64)
    active = np.flatnonzero(validity)
    if active.size > 1:
        order[active] = rng.permutation(active)
    return order


def swap_permutation(size: int, first: int, second: int) -> np.ndarray:
    order = np.arange(int(size), dtype=np.int64)
    first, second = int(first), int(second)
    if first == second or min(first, second) < 0 or max(first, second) >= size:
        raise ValueError("swap requires two distinct in-range branches")
    order[first], order[second] = order[second], order[first]
    return order


def permute_complete_bundle(
    state,
    mask,
    caps,
    action_mask,
    permutation,
    tie_break_priorities=None,
) -> dict[str, np.ndarray]:
    """Move every branch-indexed inference constraint as one bundle."""
    state = np.asarray(state)
    mask = np.asarray(mask, dtype=bool)
    caps = np.asarray(caps, dtype=np.int64)
    action_mask = np.asarray(action_mask, dtype=bool)
    if state.ndim != 2:
        raise ValueError("state must have shape [branches, features]")
    size = state.shape[0]
    if mask.shape != (size,) or caps.shape != (size,):
        raise ValueError("state, mask, and caps are not branch-aligned")
    if action_mask.ndim != 2 or action_mask.shape[0] != size:
        raise ValueError("action mask is not branch-aligned")
    expected = action_mask_from_caps(mask, caps, action_mask.shape[1])
    if not np.array_equal(action_mask, expected):
        raise ValueError("action mask is inconsistent with validity mask/caps")
    order = validate_permutation(permutation, size)
    if tie_break_priorities is None:
        priorities = None
    else:
        priorities = np.asarray(tie_break_priorities, dtype=np.int64)
        if priorities.shape != (size,):
            raise ValueError("tie-break priorities are not branch-aligned")
        if np.unique(priorities).size != size:
            raise ValueError("tie-break priorities must be unique")
    bundle = {
        "state": state[order].copy(),
        "mask": mask[order].copy(),
        "caps": caps[order].copy(),
        "action_mask": action_mask[order].copy(),
        "permutation": order.copy(),
    }
    if priorities is not None:
        bundle["tie_break_priorities"] = priorities[order].copy()
    moved_expected = action_mask_from_caps(
        bundle["mask"], bundle["caps"], action_mask.shape[1]
    )
    if not np.array_equal(bundle["action_mask"], moved_expected):
        raise RuntimeError("permuted action constraints lost bundle alignment")
    return bundle


def inverse_map_branch_values(values, permutation) -> np.ndarray:
    """Map a branch-indexed vector or matrix back to physical-node order."""
    values = np.asarray(values)
    if values.ndim < 1:
        raise ValueError("branch values must have at least one dimension")
    order = validate_permutation(permutation, values.shape[0])
    physical = np.empty_like(values)
    physical[order] = values
    return physical
