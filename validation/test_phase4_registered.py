"""Structural tests for the preregistered Phase 4 evaluator."""

from __future__ import annotations

from experiments.run_phase4_registered import (
    BASELINE_FACTORIES,
    BUDGETS,
    REGISTERED_SEEDS,
    TRAINING_SEEDS,
    expected_keys,
    holm_adjust,
)


def test_registered_task_matrix_has_720_unique_runs():
    specs = [
        {
            "policy_arm": f"hta_shared_b{budget}",
            "architecture": "shared_branching",
            "budget": budget,
            "training_seed": seed,
        }
        for budget in BUDGETS
        for seed in TRAINING_SEEDS
    ]
    specs.extend(
        {
            "policy_arm": "hta_independent_b12",
            "architecture": "independent_dqns",
            "budget": 12,
            "training_seed": seed,
        }
        for seed in TRAINING_SEEDS
    )
    keys = expected_keys(REGISTERED_SEEDS, specs)
    assert len(BASELINE_FACTORIES) == 6
    assert len(specs) == 18
    assert len(keys) == 30 * 24 == 720


def test_holm_adjustment_is_monotone_in_sorted_raw_p_values():
    payloads = [
        {"p_value_raw": 0.03},
        {"p_value_raw": 0.01},
        {"p_value_raw": 0.20},
    ]
    holm_adjust(payloads)
    ordered = sorted(payloads, key=lambda row: row["p_value_raw"])
    assert [row["p_value_holm"] for row in ordered] == [0.03, 0.06, 0.20]
    assert all(row["reject_holm_0_05"] is (index == 0) for index, row in enumerate(ordered))
