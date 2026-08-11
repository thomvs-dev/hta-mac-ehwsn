from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.ch_depletion_risk import ch_depletion_risk, validate_ch_risk_config
from validation.step3_pretraining_preflight import evaluate_step3_gates


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "step3_ch_role_depletion_risk_v1.json").read_text())


def test_risk_is_inactive_above_frozen_reserve_threshold():
    validate_ch_risk_config(CONFIG)
    result = ch_depletion_risk(
        CONFIG, reserve_fraction=0.5, forecast_harvest_j=0.0,
        distance_to_bs_m=70.0, intended_delivered_packets=16, frame_slot_budget=16,
    )
    assert result["raw_penalty"] == 0.0


def test_risk_targets_depleted_loaded_scheduled_ch_without_future_leakage():
    result = ch_depletion_risk(
        CONFIG, reserve_fraction=0.1, forecast_harvest_j=0.0,
        distance_to_bs_m=70.0, intended_delivered_packets=16, frame_slot_budget=16,
    )
    assert result["raw_penalty"] < 0.0
    assert result["scheduled_ch_role"] is True
    assert CONFIG["uses_realized_future_harvest"] is False


def test_step3_preflight_rejects_dominating_risk():
    row = {
        "steps": 1200, "t_fnd": 1100,
        "raw_terms": {"deaths": -1, "ch_depletion_risk": -2},
        "weighted_terms": {"deaths": -1, "ch_depletion_risk": -9},
        "qos_constraint": {
            "cumulative_counts": {"delivered": 1, "demand": 2},
            "metric_contract": {
                "ratio_scope": "episode_cumulative_target_backlog_service",
                "demand_field": "target_packets_offered",
                "fairness_metric_name": "target_cluster_service_fairness",
            },
        },
    }
    gates = evaluate_step3_gates([row], CONFIG, 1200)
    assert gates["ch_risk_activated"]["pass"]
    assert not gates["ch_risk_non_dominating"]["pass"]
