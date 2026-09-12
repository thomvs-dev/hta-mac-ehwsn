from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.architectures import (
    EquivariantSetBranchingC51,
    WorkloadConditionedEquivariantSetBranchingC51,
    warmstart_workload_conditioned,
)
from envs.step3_lifetime_env import RoleSeparatedScheduledMACEnv, configure_step3_risk
from envs.step3_policy_observation import (
    CH_CONTEXT_NAMES,
    STEP3_WORKLOAD_CONTEXT_SCHEMA,
    WORKLOAD_CONTEXT_NAMES,
)
from envs.step3_v3_env import Step3V3DynamicClusterTrainingEnv


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "paper_aligned_hasani2025_b16_qos_repaired.json"
RISK = json.loads((ROOT / "config" / "step3_v3_risk_weight_1.json").read_text())


def build_one(monkeypatch):
    configure_step3_risk(RISK)
    monkeypatch.setattr(trainer, "ScheduledIntraClusterMACEnv", RoleSeparatedScheduledMACEnv)
    monkeypatch.setattr(trainer, "DynamicClusterTrainingEnv", Step3V3DynamicClusterTrainingEnv)
    envs, _, _ = trainer.build_curriculum(
        [3950], 8,
        observation_schema=STEP3_WORKLOAD_CONTEXT_SCHEMA,
        environment_profile=PROFILE,
    )
    return envs[0]


def test_workload_schema_adds_broadcast_observable_context(monkeypatch):
    env = build_one(monkeypatch)
    observation, mask, _ = env.reset()
    layout = env.observation_layout
    context_features = len(CH_CONTEXT_NAMES) + len(WORKLOAD_CONTEXT_NAMES)
    context_start = layout["embedding_start"] - context_features
    assert observation.shape == (100, 72)
    assert layout["schema"] == STEP3_WORKLOAD_CONTEXT_SCHEMA
    assert layout["scheduled_ch_context_features"] == 14
    assert np.allclose(
        observation[:, context_start:layout["embedding_start"]],
        observation[0, context_start:layout["embedding_start"]],
        atol=0.0,
    )
    assert np.all((observation[:, context_start:layout["embedding_start"]] >= 0.0))
    assert np.all((observation[:, context_start:layout["embedding_start"]] <= 1.0))
    assert np.any(mask)


def test_workload_ewma_updates_from_observed_demand(monkeypatch):
    env = build_one(monkeypatch)
    _, mask, _ = env.reset()
    before = env.step3_demand_ewma
    action = np.zeros(env.base.n_nodes, dtype=np.int64)
    active = np.flatnonzero(mask)
    action[active[: min(len(active), env.base.cfg.frame_slot_budget)]] = 1
    _, _, _, info = env.step(action)
    expected = 0.8 * before + 0.2 * (
        info["target_packets_offered"] / max(1, len(info["target_members"]))
    )
    assert np.isfinite(env.step3_demand_ewma)
    assert env.step3_demand_ewma == expected


def test_workload_conditioned_architecture_remains_permutation_equivariant(monkeypatch):
    env = build_one(monkeypatch)
    observation, mask, _ = env.reset()
    network = WorkloadConditionedEquivariantSetBranchingC51(
        input_dim=observation.shape[1], budget=24, max_branches=100,
        workload_start=33, workload_features=7,
    ).eval()
    state = torch.as_tensor(observation[None], dtype=torch.float32)
    valid = torch.as_tensor(mask[None], dtype=torch.bool)
    permutation = torch.randperm(observation.shape[0])
    inverse = torch.argsort(permutation)
    with torch.no_grad():
        reference = network.q_values(state, valid)
        permuted = network.q_values(state[:, permutation], valid[:, permutation])[:, inverse]
    assert torch.max(torch.abs(reference - permuted)).item() <= 2e-6


def test_workload_network_warmstart_exactly_preserves_v3_q_values(monkeypatch):
    env = build_one(monkeypatch)
    observation, mask, _ = env.reset()
    old_state = np.concatenate((observation[:, :33], observation[:, 40:]), axis=1)
    torch.manual_seed(17)
    old = EquivariantSetBranchingC51(
        input_dim=65, budget=24, max_branches=100
    ).eval()
    new = WorkloadConditionedEquivariantSetBranchingC51(
        input_dim=72, budget=24, max_branches=100,
        workload_start=33, workload_features=7,
    ).eval()
    warmstart_workload_conditioned(new, old.state_dict())
    old_tensor = torch.as_tensor(old_state[None], dtype=torch.float32)
    new_tensor = torch.as_tensor(observation[None], dtype=torch.float32)
    valid = torch.as_tensor(mask[None], dtype=torch.bool)
    with torch.no_grad():
        old_q = old.q_values(old_tensor, valid)
        new_q = new.q_values(new_tensor, valid)
    assert torch.max(torch.abs(old_q - new_q)).item() <= 1e-6
