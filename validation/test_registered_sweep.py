"""Tests for the frozen Phase 2 registered sweep plan."""

from experiments.run_phase2_registered_sweep import registered_runs


def test_registered_sweep_has_five_budgets_three_seeds_and_ablation():
    runs = registered_runs()
    shared = [row for row in runs if row["architecture"] == "shared_branching"]
    independent = [row for row in runs if row["architecture"] == "independent_dqns"]
    assert len(runs) == 18
    assert {(row["budget"], row["optimizer_seed"]) for row in shared} == {
        (budget, seed)
        for budget in (8, 12, 16, 20, 24)
        for seed in (2299, 3299, 4299)
    }
    assert {(row["budget"], row["optimizer_seed"]) for row in independent} == {
        (12, seed) for seed in (2299, 3299, 4299)
    }