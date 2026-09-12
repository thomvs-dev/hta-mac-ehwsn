from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from agents.architectures import EquivariantSetBranchingC51
from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent
from agents.budget_projection import (
    project_slot_budget,
    projection_optimality_diagnostic,
    solve_slot_budget_exact,
)
from experiments.evaluate_publication_ablation_and_all_cluster import (
    analyze_load_robust_fair_energy,
    jain_weighted_goodput_efficiency,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exact_budget_solver_matches_brute_force_for_nonconcave_values():
    q = np.asarray([
        [0.0, 0.2, 4.0, 4.1],
        [0.0, 1.5, 1.6, 1.7],
        [0.0, 1.4, 2.7, 2.8],
    ])
    caps = np.asarray([3, 2, 3])
    budget = 4
    exact, objective = solve_slot_budget_exact(q, budget, caps=caps)
    candidates = [
        action for action in itertools.product(range(4), repeat=3)
        if sum(action) <= budget and np.all(np.asarray(action) <= caps)
    ]
    brute = max(candidates, key=lambda action: sum(q[i, action[i]] for i in range(3)))
    brute_value = sum(q[i, brute[i]] for i in range(3))
    assert np.isclose(objective, brute_value)
    assert int(exact.sum()) <= budget


def test_projection_diagnostic_detects_nonconcave_greedy_regret():
    q = np.asarray([[0.0, 0.1, 5.0], [0.0, 2.0, 2.1]])
    greedy = project_slot_budget(q, 2)
    diagnostic = projection_optimality_diagnostic(q, greedy, 2)
    assert diagnostic["absolute_regret"] > 0.0
    assert diagnostic["allocation_match"] is False


def test_feature_ablation_is_applied_after_normalization():
    config = BranchingAgentConfig(
        input_dim=65, architecture="equivariant_set_branching",
        normalize_input_blocks=True, embedding_start_dim=33,
        hybrid_harvest_max_j=1.0, ablated_feature_indices=tuple(range(26, 33)),
    )
    agent = BranchingDQNAgent(config)
    state = torch.ones((1, 4, 65), dtype=torch.float32)
    transformed = agent._transform_state_tensor(state)
    assert torch.count_nonzero(transformed[..., 26:33]) == 0


def test_no_set_context_diagnostic_is_permutation_equivariant():
    torch.manual_seed(11)
    model = EquivariantSetBranchingC51(input_dim=8, hidden_dim=16, max_branches=6)
    state = torch.randn(1, 6, 8)
    mask = torch.tensor([[True, True, False, True, True, False]])
    permutation = torch.tensor([3, 0, 5, 1, 4, 2])
    reference = model.q_values_without_set_context(state, mask)
    permuted = model.q_values_without_set_context(state[:, permutation], mask[:, permutation])
    assert torch.allclose(permuted, reference[:, permutation], rtol=0.0, atol=1e-6)


def test_publication_extension_contract_keeps_evaluation_cohort_independent():
    path = ROOT / "config" / "publication_ablation_all_cluster_v1.json"
    contract = load_contract(path)
    assert not set(contract["evaluation_seeds"]) & set(contract["prior_used_seeds"])
    assert len(contract["evaluation_seeds"]) == 20
    assert contract["horizon"] >= 3000
    metric = contract["derived_metrics"]["load_robust_fair_energy_retention"]
    assert metric["confirmation_cohort"] == contract["evaluation_seeds"]


def test_jain_weighted_goodput_retains_units_and_penalizes_unfairness():
    assert jain_weighted_goodput_efficiency(300.0, 1.0) == 300.0
    assert jain_weighted_goodput_efficiency(300.0, 0.8) == 240.0


def test_load_robustness_is_paired_by_seed_and_ranked_without_weight_tuning():
    contract = {
        "bootstrap_resamples": 100,
        "policies": ["hta_mac", "energy_proportional"],
        "derived_metrics": {
            "load_robust_fair_energy_retention": {
                "reference_scenario": "reference_100", "stress_scenario": "traffic_high"
            }
        },
    }
    raw = {
        "reference_100": {
            "hta_mac": {"1": {"jain_weighted_goodput_per_j": 100.0}, "2": {"jain_weighted_goodput_per_j": 200.0}},
            "energy_proportional": {"1": {"jain_weighted_goodput_per_j": 100.0}, "2": {"jain_weighted_goodput_per_j": 200.0}},
        },
        "traffic_high": {
            "hta_mac": {"1": {"jain_weighted_goodput_per_j": 95.0}, "2": {"jain_weighted_goodput_per_j": 190.0}},
            "energy_proportional": {"1": {"jain_weighted_goodput_per_j": 80.0}, "2": {"jain_weighted_goodput_per_j": 160.0}},
        },
    }
    result = analyze_load_robust_fair_energy(
        contract, raw, [1, 2], np.random.default_rng(4)
    )
    assert result["rank_by_mean_retention"] == ["hta_mac", "energy_proportional"]
    assert np.isclose(result["policies"]["hta_mac"]["mean_retention"], 0.95)
