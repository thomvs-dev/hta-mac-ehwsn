import json
from pathlib import Path

import numpy as np

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.state_conditioned_admission_teacher import (
    StateConditionedAdmissionConfig,
    state_conditioned_admission_action,
)
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from experiments.evaluate_step4_publication_evidence import build_transfer_environments


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/v2_phase_b8_state_conditioned_admission_20260904.json"


def _config(contract):
    return StateConditionedAdmissionConfig.from_payload({
        "allocator": contract["fixed_allocator"],
        "reserve_fraction": 0.005,
        "maximum_slot_reduction": 4,
        "minimum_service_cap": 20,
    })


def test_state_conditioned_action_is_deterministic_feasible_and_nondegrading():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    runtime = dict(contract)
    runtime["development_seeds"] = [13000]
    runtime["observation_schema"] = STEP3_CH_CONTEXT_SCHEMA
    env = build_transfer_environments(runtime, contract["scenario"], trace=None)[0]
    observation, mask, _ = env.reset()
    _, active, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
    action_a, audit_a = state_conditioned_admission_action(
        env, active, caps, contract["physical_budget"], _config(contract), return_audit=True
    )
    action_b, audit_b = state_conditioned_admission_action(
        env, active, caps, contract["physical_budget"], _config(contract), return_audit=True
    )
    assert np.array_equal(action_a, action_b)
    assert audit_a == audit_b
    assert np.all(action_a >= 0) and np.all(action_a <= caps)
    assert np.all(action_a[~active] == 0)
    assert int(action_a.sum()) <= contract["physical_budget"]
    assert audit_a["selected_served_expiring"] >= audit_a["full_served_expiring"]
    assert audit_a["selected_projected_fairness"] + 1e-12 >= audit_a["full_projected_fairness"]


def test_b8_contract_has_fresh_disjoint_cohorts_and_exact_gate():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    unavailable = {
        seed for first, last in contract["unavailable_seed_ranges"] for seed in range(first, last + 1)
    }
    development, reserved = set(contract["development_seeds"]), set(contract["reserved_gate_seeds"])
    assert len(development) == 10 and len(reserved) == 20
    assert development.isdisjoint(reserved | unavailable)
    assert reserved.isdisjoint(unavailable)
    assert set(range(3900, 3920)) <= unavailable
    assert contract["gate_targets_unchanged"] == {
        "delivery_min": 0.450,
        "packets_per_j_min": 244.8,
        "stale_ratio_max": 0.035,
        "fairness_min": 0.92,
        "rmst_min": 128.0,
        "feasibility_violations_max": 0,
    }
    assert contract["decision_rules"]["no_neural_training"] is True
