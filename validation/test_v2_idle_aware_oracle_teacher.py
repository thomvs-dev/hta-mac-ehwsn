import json
from pathlib import Path

import numpy as np

import experiments.train_phase2_dynamic_curriculum as trainer
from agents.idle_aware_oracle_teacher import (
    IdleAwareOracleTeacherConfig,
    idle_aware_oracle_action,
)
from envs.step3_policy_observation import STEP3_CH_CONTEXT_SCHEMA
from experiments.evaluate_step4_publication_evidence import build_transfer_environments


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/v2_phase_b6_idle_aware_teacher_development_20260904.json"


def _payload():
    return {
        "base_teacher": {
            "energy_weight": 4.0,
            "depletion_weight": 0.5,
            "fairness_guard": 0.9,
            "stale_guard": 0.035,
            "reserve_floor": 0.08,
        },
        "reduced_member_limit": 8,
    }


def test_action_is_deterministic_feasible_and_constraint_preserving():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    runtime = dict(contract)
    runtime["development_seeds"] = [12100]
    runtime["observation_schema"] = STEP3_CH_CONTEXT_SCHEMA
    env = build_transfer_environments(runtime, contract["scenario"], trace=None)[0]
    observation, mask, _ = env.reset()
    _, active, caps = trainer.padded_state(env, observation, mask, env.base.n_nodes)
    config = IdleAwareOracleTeacherConfig.from_payload(_payload())
    action_a, audit_a = idle_aware_oracle_action(
        env, active, caps, contract["budget"], config, return_audit=True
    )
    action_b, audit_b = idle_aware_oracle_action(
        env, active, caps, contract["budget"], config, return_audit=True
    )
    assert np.array_equal(action_a, action_b)
    assert np.all(action_a >= 0)
    assert np.all(action_a <= caps)
    assert np.all(action_a[~active] == 0)
    assert int(action_a.sum()) == min(contract["budget"], int(caps[active].sum()))
    assert audit_a["predicted_reduced_energy_saving_j"] >= 0.0
    assert audit_a == audit_b
    oracle = audit_a["oracle"]
    assert oracle["oracle_expiring"] >= oracle["base_expiring"]
    assert oracle["oracle_fairness"] + 1e-12 >= oracle["base_fairness"]
    assert oracle["oracle_minimum_post_energy"] + 1e-15 >= oracle["base_minimum_post_energy"]


def test_contract_keeps_exact_gate_and_seed_boundaries():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["gate_targets_unchanged"] == {
        "delivery_min": 0.450,
        "packets_per_j_min": 244.8,
        "stale_ratio_max": 0.035,
        "fairness_min": 0.92,
        "rmst_min": 128.0,
        "feasibility_violations_max": 0,
    }
    development = set(contract["development_seeds"])
    reserved = set(contract["reserved_gate_seeds"])
    unavailable = set(contract["unavailable_seeds"])
    assert development <= unavailable
    assert development.isdisjoint(reserved)
    assert reserved.isdisjoint(unavailable)
    assert set(range(3900, 3920)) <= unavailable
    assert set(range(3900, 3920)).isdisjoint(development | reserved)
    assert contract["decision_rules"]["no_neural_training"] is True
