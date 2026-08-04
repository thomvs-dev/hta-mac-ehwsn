import numpy as np
import pytest

from agents.branch_permutation import (
    action_mask_from_caps,
    active_branch_permutation,
    inverse_map_branch_values,
    permute_complete_bundle,
    swap_permutation,
)


def test_complete_bundle_moves_state_mask_caps_and_action_mask_together():
    state = np.arange(12, dtype=np.float32).reshape(4, 3)
    mask = np.array([True, False, True, True])
    caps = np.array([3, 0, 1, 2])
    action_mask = action_mask_from_caps(mask, caps, actions=4)
    order = np.array([2, 0, 3, 1])

    moved = permute_complete_bundle(state, mask, caps, action_mask, order)

    np.testing.assert_array_equal(moved["state"], state[order])
    np.testing.assert_array_equal(moved["mask"], mask[order])
    np.testing.assert_array_equal(moved["caps"], caps[order])
    np.testing.assert_array_equal(moved["action_mask"], action_mask[order])
    np.testing.assert_array_equal(
        moved["action_mask"],
        action_mask_from_caps(moved["mask"], moved["caps"], actions=4),
    )


def test_inconsistent_naively_unpermuted_caps_or_mask_is_rejected():
    state = np.zeros((3, 2), dtype=np.float32)
    mask = np.array([True, False, True])
    caps = np.array([3, 0, 1])
    correct = action_mask_from_caps(mask, caps, actions=4)

    with pytest.raises(ValueError, match="inconsistent"):
        permute_complete_bundle(
            state,
            mask,
            np.array([1, 0, 3]),
            correct,
            np.array([2, 1, 0]),
        )


def test_inverse_mapping_restores_physical_node_identity():
    physical = np.array([0, 1, 2, 3])
    order = swap_permutation(4, 0, 3)
    branch_output = physical[order]
    np.testing.assert_array_equal(
        inverse_map_branch_values(branch_output, order), physical
    )
    assert not np.array_equal(branch_output, physical)


def test_active_permutation_never_moves_inactive_padding():
    mask = np.array([True, False, True, False, True])
    order = active_branch_permutation(mask, np.random.default_rng(17))
    np.testing.assert_array_equal(order[~mask], np.flatnonzero(~mask))
    np.testing.assert_array_equal(np.sort(order[mask]), np.flatnonzero(mask))
