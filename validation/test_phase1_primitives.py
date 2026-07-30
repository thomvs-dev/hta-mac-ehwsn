"""Focused regression tests for Phase 1 accounting and projection."""

from __future__ import annotations

import numpy as np

from agents.budget_projection import project_slot_budget
from core.energy.idle_model import energy_update, idle_listening_energy
from core.hmm.rectified_moments import rectified_gaussian_moments


def test_idle_energy_uses_inherited_electronics_constant():
    result = idle_listening_energy(
        np.array([0, 1, 3]),
        p_idle_j_per_bit_time=5.0e-8,
        slot_bit_times=4000,
    )
    np.testing.assert_allclose(result, [0.0, 2.0e-4, 6.0e-4])


def test_energy_update_clamps_at_zero_and_capacity():
    result = energy_update(
        [0.1, 0.4], consumed=[0.2, 0.0], harvested=[0.0, 0.2], capacity=0.5
    )
    np.testing.assert_allclose(result, [0.0, 0.5])


def test_projection_respects_predecessor_levels_and_budget():
    q = np.array(
        [
            [0.0, 2.0, 2.5, 2.6],
            [0.0, 1.5, 3.0, 3.1],
        ]
    )
    allocation = project_slot_budget(q, budget=3)
    np.testing.assert_array_equal(allocation, [1, 2])
    assert int(allocation.sum()) == 3


def test_projection_may_leave_negative_gain_budget_unused():
    allocation = project_slot_budget([[1.0, 0.0, -1.0]], budget=2)
    np.testing.assert_array_equal(allocation, [0])


def test_rectified_zero_mean_moments_match_closed_form():
    mean, second = rectified_gaussian_moments([0.0], [1.0], scale=2.0)
    np.testing.assert_allclose(mean, [2.0 / np.sqrt(2.0 * np.pi)])
    np.testing.assert_allclose(second, [2.0])


def test_data_and_control_idle_variants_have_expected_ratio():
    data = idle_listening_energy(
        1, p_idle_j_per_bit_time=5.0e-8, slot_bit_times=4000
    )
    control = idle_listening_energy(
        1, p_idle_j_per_bit_time=5.0e-8, slot_bit_times=100
    )
    np.testing.assert_allclose(data / control, 40.0)
