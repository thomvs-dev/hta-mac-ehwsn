"""Regression tests for the post-repair paper-aligned QoS contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agents.qos_constraints import QoSConstraintConfig, QoSConstraintController
from envs.policy_observation import PHASE2D_POLICY_SCHEMA
from experiments.train_phase2_dynamic_curriculum import build_curriculum


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "paper_aligned_hasani2025_b16_qos_repaired.json"
QOS = ROOT / "config" / "paper_aligned_hasani2025_qos_constraints_repaired.json"


def test_repaired_qos_uses_same_step_offered_backlog_without_clipping():
    payload = json.loads(QOS.read_text(encoding="utf-8"))
    controller = QoSConstraintController(QoSConstraintConfig.from_payload(payload))
    penalty, details = controller.evaluate(
        delivered=7, offered=10, stale=2, fairness=0.8
    )
    assert penalty == pytest.approx(0.0)
    assert details["ratios"]["delivery"] == pytest.approx(0.7)
    assert details["ratios"]["stale"] == pytest.approx(0.2)
    assert details["metric_contract"]["demand_field"] == "target_packets_offered"
    with pytest.raises(ValueError, match="cannot exceed"):
        controller.evaluate(delivered=2, offered=1, stale=0, fairness=1.0)


def test_repaired_dynamic_environment_emits_coherent_service_cohort():
    environments, _, _ = build_curriculum(
        [2400],
        8,
        observation_schema=PHASE2D_POLICY_SCHEMA,
        environment_profile=PROFILE,
    )
    env = environments[0]
    _, mask, _ = env.reset()
    for _ in range(5):
        action = np.zeros(env.base.n_nodes, dtype=np.int64)
        active = np.flatnonzero(mask)
        action[active[: min(len(active), env.base.cfg.frame_slot_budget)]] = 1
        _, mask, done, info = env.step(action)
        assert info["target_packets_delivered"] <= info["target_packets_offered"]
        assert info["target_stale_drops"] <= info["target_packets_offered"]
        assert info["target_cluster_service_fairness"] == pytest.approx(
            info["reward_raw_terms"]["queue_fairness"]
        )
        assert int(info["target_packets_delivered_per_node"].sum()) == info[
            "target_packets_delivered"
        ]
        assert int(info["target_packets_offered_per_node"].sum()) == info[
            "target_packets_offered"
        ]
        if done:
            break


def test_repaired_seed_sets_are_fresh_and_nonleaky():
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    assert profile["optimizer_seeds"] == [5399, 6399, 7399]
    assert set(profile["optimizer_seeds"]).isdisjoint([5299, 6299, 7299])
    assert set(profile["development_seeds"]).isdisjoint(
        profile["reserved_confirmation_seeds"]
    )
    assert set(profile["development_seeds"]).isdisjoint(
        profile["prohibited_registered_held_out_seeds"]
    )
