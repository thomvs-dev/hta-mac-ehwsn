from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

import agents.reward_model as reward_module
import experiments.train_phase2_dynamic_curriculum as trainer
import experiments.train_step3_v3 as v3_trainer
from agents.branching_dqn import BranchingAgentConfig
from agents.qos_constraints_v3 import Step3QoSConstraintConfig, Step3QoSConstraintController
from envs.step3_lifetime_env import RoleSeparatedScheduledMACEnv, configure_step3_risk
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from envs.step3_v3_env import Step3V3DynamicClusterTrainingEnv, episode_service_fairness


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "paper_aligned_hasani2025_b16_qos_repaired.json"
RISK = json.loads((ROOT / "config" / "step3_v3_risk_weight_1.json").read_text())
QOS = json.loads((ROOT / "config" / "step3_v3_qos_ema_floor_candidate.json").read_text())


def build_one(monkeypatch):
    configure_step3_risk(RISK)
    monkeypatch.setattr(trainer, "ScheduledIntraClusterMACEnv", RoleSeparatedScheduledMACEnv)
    monkeypatch.setattr(trainer, "DynamicClusterTrainingEnv", Step3V3DynamicClusterTrainingEnv)
    envs, manifest, _ = trainer.build_curriculum(
        [2400], 5, observation_schema=STEP3_CH_CONTEXT_SCHEMA,
        environment_profile=PROFILE,
    )
    return envs[0], manifest


def test_v3_broadcasts_observable_ch_context_without_changing_schedule(monkeypatch):
    env, manifest = build_one(monkeypatch)
    observation, mask, _ = env.reset()
    layout = env.observation_layout
    assert observation.shape == (100, 65)
    assert layout["embedding_start"] == 33
    assert layout["scheduled_ch_context_features"] == 7
    assert np.allclose(observation[:, 26:33], observation[0, 26:33], atol=0.0)
    assert manifest[0]["schedule_schema_version"] == "paper_aligned_exogenous_leach_v1"
    assert np.any(mask)


def test_v3_ch_reserve_changes_context_without_changing_member_local_rows(monkeypatch):
    env, _ = build_one(monkeypatch)
    before, _, _ = env.reset()
    ch = int(env.ch)
    env.base.energy[ch] *= 0.5
    after = env._observation(env.base._state())
    assert np.allclose(after[:, 26], before[:, 26] * 0.5, atol=1e-6)
    non_ch = np.arange(env.base.n_nodes) != ch
    assert np.allclose(after[non_ch, :26], before[non_ch, :26], atol=1e-6)
    assert np.allclose(after[:, 33:], before[:, 33:], atol=1e-6)


def test_v3_qos_controller_enforces_floors_and_ema():
    config = Step3QoSConstraintConfig.from_payload(QOS)
    controller = Step3QoSConstraintController(config)
    assert controller.multipliers["delivery"] >= 2.0
    controller.begin_episode()
    for _ in range(10):
        controller.evaluate(delivered=1, offered=10, stale=0, fairness=0.5)
    update = controller.end_episode()
    assert update["update_variant"] == "ema_episode_end"
    assert update["multipliers"]["delivery"] > 5.0
    assert update["multipliers"]["fairness"] >= 1.0


def test_episode_service_fairness_is_cohort_consistent_and_empty_safe():
    assert episode_service_fairness([0, 0], [0, 0]) == 1.0
    assert episode_service_fairness([5, 10, 0], [10, 20, 0]) == pytest.approx(1.0)
    assert episode_service_fairness([10, 0], [10, 10]) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="equal shape"):
        episode_service_fairness([1], [1, 2])


def test_v3_exposes_prospective_episode_service_fairness(monkeypatch):
    env, _ = build_one(monkeypatch)
    _, mask, _ = env.reset()
    action = np.zeros(env.base.n_nodes, dtype=np.int64)
    active = np.flatnonzero(mask)
    action[active[: min(len(active), env.base.cfg.frame_slot_budget)]] = 1
    _, _, _, info = env.step(action)
    assert 0.0 <= info["target_episode_service_fairness"] <= 1.0
    assert info["target_episode_service_fairness"] == pytest.approx(
        env.step3_qos_counts["episode_service_fairness"]
    )


def test_v3_reward_balance_includes_ch_risk():
    v3_trainer._RISK_CONFIG = RISK
    trainer.TERM_ORDER = tuple(reward_module.TERM_ORDER) + ("ch_depletion_risk",)
    rows = [{"weighted_terms": {name: 1.0 for name in trainer.TERM_ORDER}}]
    totals, fractions, _ = v3_trainer.contribution_balance(rows)
    assert totals["ch_depletion_risk"] == 1.0
    assert fractions["ch_depletion_risk"] == pytest.approx(1 / len(trainer.TERM_ORDER))


def test_stability_checkpoint_exports_atomically():
    test_root = ROOT / "outputs" / "validation_artifacts" / f"step3_v3_atomic_{uuid4().hex}"
    v3_trainer._EXPORT_DIR = test_root / "drive"
    v3_trainer._OPTIMIZER_SEED = 5599
    v3_trainer._RUN_NAME = "test"
    agent = v3_trainer.RecoveryExportAgent(
        BranchingAgentConfig(
            input_dim=65, actions=4, budget=16, max_branches=100,
            architecture="equivariant_set_branching",
            state_schema=STEP3_CH_CONTEXT_SCHEMA,
            embedding_start_dim=33,
        )
    )
    local = test_root / "local"
    local.mkdir(parents=True)
    (local / "episodes.jsonl").write_text('{"episode": 500}\n')
    agent.save(local / "stability_episode_500.pt", {"episode": 500})
    exported = v3_trainer._EXPORT_DIR / "stability_episode_500.pt"
    assert exported.is_file()
    assert (v3_trainer._EXPORT_DIR / "episodes.jsonl").is_file()
    sidecar = json.loads((v3_trainer._EXPORT_DIR / "stability_episode_500.recovery.json").read_text())
    assert sidecar["finalization_recoverable"] is True
    assert sidecar["checkpoint_sha256"] == v3_trainer.sha256(exported)
