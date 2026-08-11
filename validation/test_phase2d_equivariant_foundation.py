"""Correctness gates for the Phase 2D equivariant policy foundation."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from agents.architectures import EquivariantSetBranchingC51
from agents.branch_permutation import (
    action_mask_from_caps,
    inverse_map_branch_values,
    permute_complete_bundle,
)
from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent
from agents.budget_projection import project_slot_budget
from envs.policy_observation import (
    PHASE2D_POLICY_SCHEMA,
    action_validity_features,
    build_policy_observation,
    packet_age_histogram,
    policy_feature_layout,
)


def _fake_base():
    return SimpleNamespace(
        n_nodes=3,
        cfg=SimpleNamespace(
            packet_ttl_rounds=3,
            queue_max_packets=5,
            n_max=3,
        ),
        packet_ages=[[0, 0, 1], [3, 3, 3], []],
        queue=np.array([3, 3, 0], dtype=np.int64),
        embedding=np.arange(96, dtype=np.float32).reshape(3, 32),
    )


def test_phase2d_observation_exposes_ttl_age_and_action_validity():
    base = _fake_base()
    physical = np.zeros((3, 18), dtype=np.float32)
    physical[:, 5] = base.queue / base.cfg.queue_max_packets
    active = np.array([True, True, False])

    observation = build_policy_observation(
        base, physical, active, schema=PHASE2D_POLICY_SCHEMA
    )
    layout = policy_feature_layout(base, PHASE2D_POLICY_SCHEMA)

    assert observation.shape == (3, 58)
    assert layout == {
        "schema": PHASE2D_POLICY_SCHEMA,
        "physical_features": 18,
        "packet_age_features": 4,
        "action_validity_features": 4,
        "embedding_features": 32,
        "embedding_start": 26,
        "total_features": 58,
    }
    np.testing.assert_allclose(
        observation[0, 18:22], [0.4, 0.2, 0.0, 0.0]
    )
    np.testing.assert_allclose(
        observation[1, 18:22], [0.0, 0.0, 0.0, 0.6]
    )
    np.testing.assert_array_equal(observation[0, 22:26], [1, 1, 1, 1])
    np.testing.assert_array_equal(observation[2, 22:26], [0, 0, 0, 0])
    assert not np.array_equal(observation[0], observation[1])


def test_packet_age_histogram_rejects_expired_internal_state():
    base = _fake_base()
    base.packet_ages[0] = [4]
    try:
        packet_age_histogram(base)
    except RuntimeError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("expired packet age was silently encoded")


def test_action_validity_matches_caps_and_active_mask():
    base = _fake_base()
    active = np.array([True, True, False])
    features = action_validity_features(base, active)
    np.testing.assert_array_equal(features[0], [1, 1, 1, 1])
    np.testing.assert_array_equal(features[1], [1, 1, 1, 1])
    np.testing.assert_array_equal(features[2], [0, 0, 0, 0])


def test_equivariant_network_passes_twenty_permutations():
    torch.manual_seed(2299)
    network = EquivariantSetBranchingC51(
        input_dim=58,
        hidden_dim=32,
        actions=4,
        atoms=11,
        budget=12,
        max_branches=10,
    ).eval()
    state = torch.randn(2, 10, 58)
    mask = torch.tensor(
        [
            [True, True, False, True, True, False, True, True, True, False],
            [True, False, True, True, False, True, True, False, True, True],
        ]
    )
    reference = network.q_values(state, mask)
    reference_distribution = network(state, mask)
    rng = np.random.default_rng(2299)
    maximum_error = 0.0
    for _ in range(20):
        order = torch.as_tensor(rng.permutation(10), dtype=torch.long)
        moved = network.q_values(state[:, order], mask[:, order])
        moved_distribution = network(state[:, order], mask[:, order])
        expected = reference[:, order]
        maximum_error = max(
            maximum_error,
            float(torch.max(torch.abs(moved - expected)).detach()),
            float(
                torch.max(
                    torch.abs(moved_distribution - reference_distribution[:, order])
                ).detach()
            ),
        )
    assert maximum_error <= 1e-6


def test_equivariant_network_has_cross_node_context():
    torch.manual_seed(4)
    network = EquivariantSetBranchingC51(
        input_dim=3,
        hidden_dim=16,
        actions=2,
        atoms=7,
        budget=2,
        max_branches=4,
    ).eval()
    first = torch.zeros(1, 4, 3)
    second = first.clone()
    second[:, 1, 0] = 1.0
    mask = torch.ones(1, 4, dtype=torch.bool)
    assert not torch.allclose(
        network.q_values(first, mask)[:, 0],
        network.q_values(second, mask)[:, 0],
    )


def test_equivariant_parameter_count_is_branch_capacity_independent():
    small = EquivariantSetBranchingC51(max_branches=20)
    large = EquivariantSetBranchingC51(max_branches=100)
    small_count = sum(parameter.numel() for parameter in small.parameters())
    large_count = sum(parameter.numel() for parameter in large.parameters())
    assert small_count == large_count
    assert small_count < 2_842_811


def test_tie_priority_moves_with_bundle_and_preserves_allocation():
    q_values = np.array(
        [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    )
    mask = np.ones(4, dtype=bool)
    caps = np.ones(4, dtype=np.int64)
    priorities = np.array([40, 10, 30, 20], dtype=np.int64)
    action_mask = action_mask_from_caps(mask, caps, actions=2)
    reference = project_slot_budget(
        q_values, budget=2, tie_break_priorities=priorities
    )
    np.testing.assert_array_equal(reference, [0, 1, 0, 1])

    order = np.array([2, 0, 3, 1])
    bundle = permute_complete_bundle(
        q_values,
        mask,
        caps,
        action_mask,
        order,
        tie_break_priorities=priorities,
    )
    moved = project_slot_budget(
        bundle["state"],
        budget=2,
        tie_break_priorities=bundle["tie_break_priorities"],
    )
    np.testing.assert_array_equal(
        inverse_map_branch_values(moved, order), reference
    )


def test_phase2d_agent_normalizes_embedding_only_and_learns():
    config = BranchingAgentConfig(
        input_dim=58,
        actions=4,
        budget=6,
        batch_size=2,
        replay_capacity=8,
        warmup=2,
        max_branches=5,
        architecture="equivariant_set_branching",
        state_schema=PHASE2D_POLICY_SCHEMA,
        embedding_start_dim=26,
        normalize_input_blocks=True,
        hybrid_harvest_max_j=4e-4,
    )
    agent = BranchingDQNAgent(config)
    state_tensor = torch.zeros(1, 5, 58)
    state_tensor[..., 18:26] = 0.5
    state_tensor[..., 26:] = torch.arange(32, dtype=torch.float32)
    transformed = agent._transform_state_tensor(state_tensor)
    torch.testing.assert_close(
        transformed[..., 18:26], state_tensor[..., 18:26]
    )
    torch.testing.assert_close(
        transformed[..., 26:].mean(dim=-1),
        torch.zeros(1, 5),
        atol=1e-6,
        rtol=0,
    )

    rng = np.random.default_rng(9)
    mask = np.ones(5, dtype=bool)
    caps = np.full(5, 3, dtype=np.int64)
    for index in range(2):
        state = rng.normal(size=(5, 58)).astype(np.float32)
        next_state = rng.normal(size=(5, 58)).astype(np.float32)
        action = np.ones(5, dtype=np.int64)
        agent.store(
            state,
            action,
            reward=1.0 + index,
            next_state=next_state,
            done=index == 1,
            mask=mask,
            next_mask=mask,
            caps=caps,
            next_caps=caps,
        )
    loss = agent.learn(beta=0.4)
    assert loss is not None and np.isfinite(loss)
