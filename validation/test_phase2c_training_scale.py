"""Regression tests for frozen Phase 2C reward scaling and head reset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from agents.branching_dqn import BranchingAgentConfig, BranchingDQNAgent
from experiments.train_phase2_dynamic_curriculum import (
    learning_reward,
    load_reward_scale,
    residual_energy_metrics,
    validate_reward_scale_payload,
)


def frozen_payload(scale=0.25):
    return {
        "status": "frozen_development_scale",
        "reward_scale": scale,
        "apply_to": "replay_and_c51_reward_only",
        "physical_metrics_scaled": False,
        "held_out_seeds_used": False,
        "support": {"v_min": -30.0, "v_max": 30.0, "atoms": 51},
    }


def test_learning_reward_scales_only_supplied_value():
    assert learning_reward(12.0, 0.25) == 3.0
    assert learning_reward(-4.0, 0.25) == -1.0
    with pytest.raises(ValueError, match="non-finite"):
        learning_reward(np.inf, 0.25)


def test_frozen_reward_scale_loader_records_sha_and_payload():
    scale, evidence = load_reward_scale("config/phase2c_return_scale.json")
    assert scale == pytest.approx(0.14436784678738615)
    assert Path(evidence["path"]).name == "phase2c_return_scale.json"
    assert len(evidence["sha256"]) == 64
    assert evidence["payload"]["physical_metrics_scaled"] is False


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("status", "draft", "not frozen"),
        ("apply_to", "all_metrics", "unsupported application"),
        ("physical_metrics_scaled", True, "must remain unscaled"),
        ("held_out_seeds_used", True, "must not use held-out"),
        ("reward_scale", 1.1, "must be finite"),
    ],
)
def test_reward_scale_loader_rejects_invalid_evidence(
    field, value, message
):
    payload = frozen_payload()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        validate_reward_scale_payload(payload)


def test_categorical_reset_preserves_trunk_and_resets_only_final_heads():
    torch.manual_seed(17)
    agent = BranchingDQNAgent(
        BranchingAgentConfig(
            input_dim=8,
            actions=4,
            max_branches=3,
            architecture="shared_branching",
            replay_capacity=16,
            learning_rate=1e-5,
            reward_scale=0.25,
        )
    )
    before = {
        name: tensor.detach().clone()
        for name, tensor in agent.online.state_dict().items()
    }
    agent.reinitialize_categorical_outputs(seed=91)
    after = agent.online.state_dict()

    reset_names = {
        "value.2.weight",
        "value.2.bias",
        "advantages.0.2.weight",
        "advantages.0.2.bias",
        "advantages.1.2.weight",
        "advantages.1.2.bias",
        "advantages.2.2.weight",
        "advantages.2.2.bias",
    }
    assert reset_names.issubset(after)
    assert any(not torch.equal(before[name], after[name]) for name in reset_names)
    for name in after:
        if name not in reset_names:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    for name, tensor in agent.target.state_dict().items():
        torch.testing.assert_close(tensor, after[name], rtol=0, atol=0)
    assert agent.optimizer.state == {}
    assert len(agent.replay) == 0
    assert agent.train_steps == 0


def test_reward_scale_is_serialized_in_agent_config():
    config = BranchingAgentConfig(reward_scale=0.14436784678738615)
    assert config.__dict__["reward_scale"] == pytest.approx(
        0.14436784678738615
    )

def test_residual_energy_metrics_use_alive_nodes_and_report_level():
    metrics = residual_energy_metrics(
        np.array([0.5, 0.3, 0.0, 0.1]),
        np.array([True, True, False, True]),
    )
    assert metrics["alive_nodes"] == 3
    assert metrics["mean_residual_energy_j"] == pytest.approx(0.3)
    assert metrics["min_residual_energy_j"] == pytest.approx(0.1)
    assert metrics["p10_residual_energy_j"] == pytest.approx(0.14)
    assert 0.0 < metrics["residual_energy_fairness"] <= 1.0
    assert metrics["residual_energy_cv"] > 0.0


def test_residual_energy_metrics_handle_no_alive_nodes():
    metrics = residual_energy_metrics(np.zeros(2), np.zeros(2, dtype=bool))
    assert metrics["alive_nodes"] == 0
    assert metrics["mean_residual_energy_j"] == 0.0
    assert metrics["residual_energy_fairness"] == 0.0