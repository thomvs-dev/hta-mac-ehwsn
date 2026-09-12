from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.evaluate_step4_load_adaptive_residual import (
    adaptive_removal_count,
    adaptive_upper_target,
    expanded_prohibited,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
BAND = {
    "lower_delivery_target": 0.235,
    "upper_delivery_target": 0.300,
    "reserve_floor": 0.20,
    "completion_fraction": 0.50,
}
CANDIDATE = {
    "raw_policy_retention": 0.90,
    "maximum_upper_target": 0.95,
    "energy_gate_exponent": 1.0,
}


def dummy_env(ch_energy=1.0, demand=0, delivered=0, offered=10):
    queue = np.zeros(4, dtype=np.int64)
    queue[1:] = offered // 3
    queue[1] += offered - int(queue.sum())
    return SimpleNamespace(
        ch=0,
        members=np.asarray([1, 2, 3]),
        step3_qos_counts={"demand": demand, "delivered": delivered},
        base=SimpleNamespace(
            queue=queue,
            alive=np.ones(4, dtype=bool),
            energy=np.asarray([ch_energy, 1.0, 1.0, 1.0]),
            cfg=SimpleNamespace(initial_energy_j=1.0),
        ),
    )


def test_contract_keeps_all_three_cohorts_disjoint():
    contract = load_contract(ROOT / "config" / "step4_load_adaptive_residual_development_v1.json")
    calibration = set(contract["calibration_seeds"])
    validation = set(contract["validation_seeds"])
    confirmation = set(contract["confirmation_seeds"])
    assert calibration.isdisjoint(validation | confirmation | expanded_prohibited(contract))
    assert validation.isdisjoint(confirmation | expanded_prohibited(contract))
    assert contract["confirmation_seeds_opened"] is False


def test_raw_policy_relative_target_retains_service_when_energy_is_healthy():
    env = dummy_env(ch_energy=1.0, offered=10)
    audit = adaptive_upper_target(env, np.asarray([0, 3, 3, 3]), CANDIDATE, BAND)
    assert audit["raw_instant_service_ratio"] == 0.9
    assert abs(audit["upper_target"] - 0.81) < 1e-12


def test_energy_gate_reverts_to_frozen_upper_at_reserve_floor():
    env = dummy_env(ch_energy=0.2, offered=10)
    audit = adaptive_upper_target(env, np.asarray([0, 3, 3, 3]), CANDIDATE, BAND)
    assert audit["energy_gate"] == 0.0
    assert audit["upper_target"] == BAND["upper_delivery_target"]


def test_minimum_energy_gate_retains_bounded_service_at_reserve_floor():
    env = dummy_env(ch_energy=0.2, offered=10)
    candidate = {**CANDIDATE, "minimum_energy_gate": 0.5}
    audit = adaptive_upper_target(env, np.asarray([0, 3, 3, 3]), candidate, BAND)
    assert audit["energy_gate"] == 0.5
    assert abs(audit["upper_target"] - 0.555) < 1e-12


def test_adaptive_removal_preserves_more_raw_slots_than_frozen_band():
    env = dummy_env(ch_energy=1.0, offered=10)
    action = np.asarray([0, 3, 3, 3])
    count, audit = adaptive_removal_count(action, action, env, BAND, CANDIDATE)
    frozen_upper_allowed = int(BAND["upper_delivery_target"] * 10)
    frozen_removal = int(action.sum()) - frozen_upper_allowed
    assert count < frozen_removal
    assert count == 1
    assert audit["upper_target"] == 0.81


def test_load_conditioned_bypass_directly_reduces_frozen_removals():
    env = dummy_env(ch_energy=1.0, offered=10)
    action = np.asarray([0, 3, 3, 3])
    candidate = {
        **CANDIDATE,
        "mode": "load_conditioned_removal_bypass",
        "bypass_intercept": 0.4,
        "bypass_slope": 1.0,
        "minimum_bypass": 0.0,
        "maximum_bypass": 1.0,
        "minimum_energy_factor": 1.0,
    }
    count, audit = adaptive_removal_count(action, action, env, BAND, candidate)
    assert audit["frozen_removed_slots"] == 6
    assert abs(audit["removal_bypass"] - 0.8) < 1e-12
    assert count == 1


def test_bypass_scales_current_action_not_unbounded_cumulative_excess():
    env = dummy_env(ch_energy=1.0, demand=1000, delivered=400, offered=10)
    action = np.asarray([0, 3, 3, 3])
    candidate = {
        **CANDIDATE,
        "mode": "load_conditioned_removal_bypass",
        "bypass_intercept": 0.8,
        "bypass_slope": 0.0,
        "minimum_bypass": 0.8,
        "maximum_bypass": 0.8,
        "minimum_energy_factor": 1.0,
    }
    count, audit = adaptive_removal_count(action, action, env, BAND, candidate)
    assert audit["frozen_removed_slots"] > int(action.sum())
    assert audit["currently_removable_slots"] == int(action.sum())
    assert count == 1


def test_single_slot_removal_is_bypassed_in_discrete_action_space():
    env = dummy_env(ch_energy=1.0, demand=10, delivered=4, offered=1)
    action = np.asarray([0, 1, 0, 0])
    candidate = {
        **CANDIDATE,
        "mode": "load_conditioned_removal_bypass",
        "bypass_intercept": 0.5,
        "bypass_slope": 0.0,
        "minimum_bypass": 0.5,
        "maximum_bypass": 0.5,
        "minimum_energy_factor": 1.0,
    }
    count, audit = adaptive_removal_count(action, action, env, BAND, candidate)
    assert audit["currently_removable_slots"] == 1
    assert count == 0
