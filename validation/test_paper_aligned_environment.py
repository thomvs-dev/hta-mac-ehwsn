"""Validation for the exploratory paper-aligned B16 branch."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from agents.qos_constraints import QoSConstraintConfig, QoSConstraintController
from envs.policy_observation import PHASE2D_POLICY_SCHEMA
from experiments.paper_aligned_environment import load_profile
from experiments.train_phase2_dynamic_curriculum import build_curriculum


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "config" / "paper_aligned_hasani2025_b16.json"
QOS = ROOT / "config" / "paper_aligned_hasani2025_qos_constraints.json"


def test_profile_is_explicitly_exploratory_and_nonleaky():
    profile, evidence = load_profile(PROFILE)
    assert profile["claim_boundary"] == "paper_aligned_not_a_third_party_reproduction"
    assert profile["learned_intervention"] == "hta_mac_only"
    assert profile["held_out_seeds_used"] is False
    assert set(profile["development_seeds"]).isdisjoint(
        profile["prohibited_registered_held_out_seeds"]
    )
    assert len(evidence["sha256"]) == 64


def test_profile_curriculum_has_balanced_exogenous_b16_environment():
    environments, manifest, config = build_curriculum(
        [2400],
        6,
        observation_schema=PHASE2D_POLICY_SCHEMA,
        environment_profile=PROFILE,
    )
    assert len(environments) == 20
    assert config.frame_slot_budget == 16
    assert config.bs_position_m == (50.0, 50.0)
    assert config.thermal_scale == 0.0
    assert environments[0].base.n_nodes == 100
    assert environments[0].base.idle_energy_enabled is False
    assert manifest[0]["schedule_schema_version"] == "paper_aligned_exogenous_leach_v1"
    assert manifest[0]["environment_profile"]["profile_id"].endswith("_v1")
    bundle = environments[0].bundle
    assert all(len(frame["cluster_heads"]) == 20 for frame in bundle["schedule"])
    first_epoch = np.concatenate(
        [bundle["schedule"][index]["cluster_heads"] for index in range(5)]
    )
    assert np.array_equal(np.sort(first_epoch), np.arange(100))
    assert all(np.all(frame["thermal_states"] == 0) for frame in bundle["schedule"])


def test_episode_end_qos_update_is_once_per_episode_with_fairness_warmup():
    payload = json.loads(QOS.read_text(encoding="utf-8"))
    controller = QoSConstraintController(QoSConstraintConfig.from_payload(payload))
    before = dict(controller.multipliers)
    _, early = controller.evaluate(delivered=0, generated=10, stale=5, fairness=0.0)
    assert early["signed_violations"]["fairness"] == pytest.approx(0.0)
    assert controller.multipliers == before
    for _ in range(5):
        controller.evaluate(delivered=0, generated=10, stale=5, fairness=0.5)
    update = controller.end_episode()
    assert update["updated"] is True
    assert all(controller.multipliers[name] > before[name] for name in before)
