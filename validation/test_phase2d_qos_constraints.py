"""Tests for the frozen development-only Phase 2D QoS objective."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agents.qos_constraints import QoSConstraintConfig, QoSConstraintController
from experiments.train_phase2_dynamic_curriculum import load_qos_constraints


ROOT = Path(__file__).resolve().parents[1]


def frozen_payload():
    return json.loads(
        (ROOT / "config" / "phase2d_qos_constraints.json").read_text(
            encoding="utf-8"
        )
    )


def test_constraint_controller_penalizes_violation_and_updates_multipliers():
    controller = QoSConstraintController(
        QoSConstraintConfig.from_payload(frozen_payload())
    )
    before = dict(controller.multipliers)
    penalty, details = controller.evaluate(
        delivered=1, generated=10, stale=6, fairness=0.5
    )
    assert penalty < 0.0
    assert details["ratios"] == {
        "delivery": pytest.approx(0.1),
        "stale": pytest.approx(0.6),
        "fairness": pytest.approx(0.5),
    }
    assert all(
        controller.multipliers[name] > before[name]
        for name in ("delivery", "stale", "fairness")
    )


def test_satisfied_constraint_has_no_penalty_and_relaxes_multipliers():
    controller = QoSConstraintController(
        QoSConstraintConfig.from_payload(frozen_payload())
    )
    before = dict(controller.multipliers)
    penalty, details = controller.evaluate(
        delivered=9, generated=10, stale=0, fairness=1.0
    )
    assert penalty == pytest.approx(0.0)
    assert all(value == pytest.approx(0.0) for value in details["positive_violations"].values())
    assert all(
        controller.multipliers[name] < before[name]
        for name in ("delivery", "stale", "fairness")
    )


def test_controller_state_round_trip_preserves_multiplier_and_counts():
    config = QoSConstraintConfig.from_payload(frozen_payload())
    first = QoSConstraintController(config)
    first.evaluate(delivered=2, generated=5, stale=1, fairness=0.8)
    second = QoSConstraintController(config)
    second.load_state_dict(first.state_dict())
    assert second.state_dict() == first.state_dict()
    second.begin_episode()
    assert second.state_dict()["multipliers"] == first.state_dict()["multipliers"]
    assert second.state_dict()["episode_counts"] == {
        "delivered": 0,
        "generated": 0,
        "stale": 0,
        "steps": 0,
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("status", "draft"),
        ("held_out_seeds_used", True),
        ("minimum_delivery_ratio", 1.1),
        ("maximum_multiplier", 0.0),
    ],
)
def test_invalid_or_leaky_constraint_payload_is_rejected(field, value):
    payload = copy.deepcopy(frozen_payload())
    payload[field] = value
    with pytest.raises(ValueError):
        QoSConstraintConfig.from_payload(payload)


def test_repository_constraint_file_and_feasibility_evidence_load():
    config, evidence = load_qos_constraints(
        "config/phase2d_qos_constraints.json"
    )
    assert config.minimum_delivery_ratio == pytest.approx(0.55)
    assert evidence["feasibility_evidence_path"].endswith(
        "phase2d_qos_feasibility.json"
    )
    assert len(evidence["sha256"]) == 64
