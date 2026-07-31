"""Focused unit tests for the Phase 2 Branching Dueling C51 agent."""

from __future__ import annotations

import numpy as np
import torch

from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent, BranchingDuelingC51
from agents.reward_model import RewardModel
from envs.dynamic_cluster_training_env import DynamicClusterTrainingEnv
from envs.fixed_cluster_training_env import FixedClusterTrainingEnv
from experiments.train_phase2_dynamic_curriculum import (
    policy_stability_summary,
    reset_inspection_state,
)


def test_branching_distribution_shape_and_normalization():
    network = BranchingDuelingC51(input_dim=50, actions=4, atoms=51)
    state = torch.randn(2, 7, 50)
    log_probabilities = network(state)
    assert log_probabilities.shape == (2, 7, 4, 51)
    totals = log_probabilities.exp().sum(dim=-1)
    assert torch.allclose(totals, torch.ones_like(totals), atol=1e-6)
    assert network.q_values(state).shape == (2, 7, 4)


def test_budget_projection_respects_mask_and_budget():
    agent = BranchingDQNAgent(BranchingAgentConfig(input_dim=50, actions=4, budget=5))
    q_values = np.array([
        [0.0, 5.0, 7.0, 8.0],
        [0.0, 4.0, 8.0, 9.0],
        [0.0, 100.0, 200.0, 300.0],
    ])
    allocation = agent._project(q_values, np.array([True, True, False]), fill_budget=True)
    assert allocation.shape == (3,)
    assert int(allocation.sum()) <= 5
    assert allocation[2] == 0
    assert np.all((allocation >= 0) & (allocation <= 3))


def test_budget_projection_respects_queue_caps():
    agent = BranchingDQNAgent(
        BranchingAgentConfig(input_dim=50, actions=4, budget=8)
    )
    q_values = np.array(
        [
            [0.0, 1.0, 5.0, 9.0],
            [0.0, 2.0, 6.0, 10.0],
            [0.0, 3.0, 7.0, 11.0],
        ]
    )
    caps = np.array([0, 1, 2])
    allocation = agent._project(
        q_values, np.ones(3, dtype=bool), fill_budget=True, caps=caps
    )
    assert np.all(allocation <= caps)
    assert allocation[0] == 0


def test_death_count_includes_target_cluster_head():
    assert FixedClusterTrainingEnv._death_count(
        [True, True], [False, True], True, False
    ) == 2
    assert FixedClusterTrainingEnv._death_count(
        [True, True], [True, True], False, False
    ) == 0

def test_fixed_cluster_episode_terminates_on_first_local_death():
    assert FixedClusterTrainingEnv._is_done(False, False, 1, True, True)
    assert not FixedClusterTrainingEnv._is_done(
        False, False, 0, True, True
    )

def test_dynamic_cluster_does_not_treat_reassignment_as_death():
    assert not DynamicClusterTrainingEnv._is_done(
        False, False, 0, True
    )
    assert DynamicClusterTrainingEnv._is_done(False, False, 1, True)

def test_reward_has_no_always_sleep_incentive():
    model = RewardModel(
        scales={
            "packets_delivered": 1.05,
            "idle_energy_j": 0.0768,
            "deaths": 1.0,
            "high_harvest_alignment": 0.2,
            "declining_allocation": 0.2,
            "queue_fairness": 1.0,
        },
        weights={
            "packets_delivered": 2.0,
            "idle_energy_j": 1.0,
            "deaths": 2.0,
            "high_harvest_alignment": 0.5,
            "declining_allocation": 0.5,
            "queue_fairness": 0.5,
        },
    )
    sleeping = {
        "packets_delivered": 0.0,
        "idle_energy_j": 0.0,
        "deaths": 0.0,
        "high_harvest_alignment": 0.0,
        "declining_allocation": 0.0,
        "queue_fairness": 0.0,
    }
    productive = {
        "packets_delivered": 1.0,
        "idle_energy_j": -0.05,
        "deaths": 0.0,
        "high_harvest_alignment": 0.1,
        "declining_allocation": -0.05,
        "queue_fairness": 0.9,
    }
    sleep_reward, _ = model.evaluate(sleeping)
    productive_reward, _ = model.evaluate(productive)
    assert sleep_reward == 0.0
    assert productive_reward > sleep_reward


def test_prioritized_replay_update_produces_finite_loss():
    config = BranchingAgentConfig(
        input_dim=50,
        actions=4,
        budget=6,
        batch_size=2,
        replay_capacity=8,
        warmup=2,
        target_update_steps=2,
    )
    agent = BranchingDQNAgent(config)
    rng = np.random.default_rng(9)
    mask = np.ones(5, dtype=bool)
    for index in range(2):
        state = rng.normal(size=(5, 50)).astype(np.float32)
        next_state = rng.normal(size=(5, 50)).astype(np.float32)
        action = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        agent.store(
            state, action, reward=1.0 + index, next_state=next_state,
            done=index == 1, mask=mask, next_mask=mask,
        )
    loss = agent.learn(beta=0.4)
    assert loss is not None
    assert np.isfinite(loss)

def test_greedy_action_does_not_advance_numpy_rng():
    agent = BranchingDQNAgent(
        BranchingAgentConfig(input_dim=50, actions=4, budget=4)
    )
    state = np.zeros((3, 50), dtype=np.float32)
    mask = np.ones(3, dtype=bool)
    np.random.seed(12)
    before = np.random.get_state()
    agent.act(state, mask, epsilon=0.0)
    after = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_policy_stability_requires_three_stable_snapshots():
    snapshots = []
    for episode, scale in ((400, 1.00), (450, 1.02), (500, 0.99)):
        snapshots.append(
            {
                "episode": episode,
                "evaluation": {
                    "mean_fnd_free_steps": 140.0 * scale,
                    "mean_throughput": 1000.0 * scale,
                    "mean_queue_fairness": 0.90 * scale,
                },
            }
        )
    result = policy_stability_summary(snapshots, relative_tolerance=0.10)
    assert result["assessed"]
    assert result["pass"]

    unstable = [dict(item) for item in snapshots]
    unstable[-1] = {
        "episode": 500,
        "evaluation": {
            "mean_fnd_free_steps": 70.0,
            "mean_throughput": 500.0,
            "mean_queue_fairness": 0.45,
        },
    }
    assert not policy_stability_summary(unstable, 0.10)["pass"]


def test_heart_solar_reward_taxonomy_is_exact():
    from envs.fixed_cluster_training_env import (
        SOLAR_DECLINING_STATES,
        SOLAR_HIGH_HARVEST_STATES,
    )

    assert SOLAR_HIGH_HARVEST_STATES == (5, 7)
    assert SOLAR_DECLINING_STATES == (0, 3, 6)
    assert set(SOLAR_HIGH_HARVEST_STATES).isdisjoint(SOLAR_DECLINING_STATES)


def test_trajectory_indicators_use_only_published_solar_taxonomy():
    class Base:
        solar_states = np.arange(8, dtype=np.int64)
        thermal_states = np.array([3, 3, 3, 3, 3, 3, 3, 3])

    env = FixedClusterTrainingEnv.__new__(FixedClusterTrainingEnv)
    env.base = Base()
    env.members = np.arange(8, dtype=np.int64)
    high, declining = env._trajectory_indicators()
    np.testing.assert_array_equal(high, [False, False, False, False, False, True, False, True])
    np.testing.assert_array_equal(declining, [True, False, False, True, False, False, True, False])

def test_shared_branching_has_global_cross_node_context():
    from agents.architectures import GlobalBranchingDuelingC51

    torch.manual_seed(4)
    network = GlobalBranchingDuelingC51(
        input_dim=3, hidden_dim=16, actions=2, atoms=7, max_branches=4
    )
    first = torch.zeros(1, 4, 3)
    second = first.clone()
    second[:, 1, 0] = 1.0
    mask = torch.ones(1, 4, dtype=torch.bool)
    q_first = network.q_values(first, mask)
    q_second = network.q_values(second, mask)
    assert not torch.allclose(q_first[:, 0], q_second[:, 0])


def test_independent_dqn_ablation_has_no_cross_node_context():
    from agents.architectures import IndependentDuelingC51

    torch.manual_seed(4)
    network = IndependentDuelingC51(
        input_dim=3, hidden_dim=16, actions=2, atoms=7, max_branches=4
    )
    first = torch.zeros(1, 4, 3)
    second = first.clone()
    second[:, 1, 0] = 1.0
    mask = torch.ones(1, 4, dtype=torch.bool)
    q_first = network.q_values(first, mask)
    q_second = network.q_values(second, mask)
    torch.testing.assert_close(q_first[:, 0], q_second[:, 0])
    assert not torch.allclose(q_first[:, 1], q_second[:, 1])

def test_trajectory_q_check_changes_state_on_same_node_head():
    from types import SimpleNamespace

    from experiments.train_phase2_fixed_cluster import trajectory_q_check

    transition = np.eye(8, dtype=np.float64)
    base = SimpleNamespace(
        solar=SimpleNamespace(
            transition=transition,
            mean=np.linspace(-0.04, 0.04, 8),
            variance=np.full(8, 0.0001),
        ),
        cfg=SimpleNamespace(solar_scale=0.01),
    )
    env = SimpleNamespace(
        base=base,
        _mask=lambda: np.array([False, True, True, False]),
    )
    agent = BranchingDQNAgent(
        BranchingAgentConfig(input_dim=50, max_branches=4)
    )
    result = trajectory_q_check(
        agent, env, np.zeros((4, 50), dtype=np.float32)
    )
    assert result["same_node_same_head"]
    assert result["node_index"] == 1
    assert result["differentiated"]

def test_inspection_state_is_reset_after_terminal_evaluation():
    class TerminalThenResetEnv:
        def __init__(self):
            self.terminal = True

        def reset(self):
            self.terminal = False
            observation = np.ones((3, 50), dtype=np.float32)
            mask = np.array([False, True, False])
            return observation, mask, {"reset": True}

        def _mask(self):
            return (
                np.zeros(3, dtype=bool)
                if self.terminal
                else np.array([False, True, False])
            )

    env = TerminalThenResetEnv()
    assert not env._mask().any()
    observation = reset_inspection_state(env)
    assert env._mask().any()
    assert observation.shape == (3, 50)