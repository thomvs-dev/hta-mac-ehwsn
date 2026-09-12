import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name):
    return json.loads((ROOT / "config" / name).read_text())


def test_forward_b16_secondary_name_preserves_physical_profile():
    historical = load("paper_aligned_hasani2025_b16_qos_repaired.json")
    secondary = load("b16_secondary_qos_profile_v1.json")
    assert secondary["profile_id"].startswith("b16_secondary_")
    assert "not_a_third_party_reproduction" in secondary["claim_boundary"]
    for field in ("network", "harvesting", "mac", "development_seeds"):
        assert secondary[field] == historical[field]


def test_future_seed_protocol_is_no_go_and_outcome_independent():
    protocol = load("step3_future_seed_complete_analysis_protocol_v1.json")
    assert protocol["remaining_confirmation_seeds"] == [3401, 3402, 3403, 3404]
    assert protocol["open_permission"].startswith("no_go_")
    assert protocol["conditional_omission_permitted"] is False
    assert protocol["selection_or_retuning_on_confirmation_seeds_permitted"] is False
    required = set(protocol["analysis_pipeline_mandatory_for_every_opened_seed"])
    assert "matched_band_only_no_neural_ablation" in required
    assert "common_horizon_truncation_for_all_rate_and_cumulative_endpoints" in required
    assert "complete_report_even_when_raw_confirmation_fails" in required


def test_development_sweep_excludes_every_confirmation_and_held_out_seed():
    sweep = load("step3_dev_energy_exponent_fairness_sweep_v1.json")
    assert set(sweep["development_seeds"]).isdisjoint(sweep["prohibited_seeds"])
    assert set(range(3400, 3405)).issubset(sweep["prohibited_seeds"])
    assert set(range(3100, 3105)).issubset(sweep["prohibited_seeds"])


def test_reframed_contribution_prohibits_neural_lifetime_claim():
    decision = load("step3_reframed_contribution_decision_v1.json")
    assert decision["neural_policy_role"].startswith("primary_control_quality")
    assert decision["lifetime_claim"].startswith("comparable_not_superior")
    assert "neural_component_causes_lifetime_gain" in decision["prohibited_claims"]
    assert decision["b16_campaign_seed_decision"].startswith("do_not_open_3401_3404")


def test_primary_direction_check_is_noninferential_and_seed_safe():
    contract = load("step3_primary_idle_on_hybrid_direction_seed2400_v1.json")
    assert contract["seed"] == 2400
    assert contract["seed"] not in contract["prohibited_seeds"]
    assert set(range(3400, 3405)).issubset(contract["prohibited_seeds"])
    assert set(range(3100, 3105)).issubset(contract["prohibited_seeds"])
    assert contract["retraining_or_retuning_permitted"] is False
    assert contract["inferential_claim_permitted"] is False
    assert contract["primary_regime"]["idle_listening"] is True
    assert contract["primary_regime"]["thermal_hmm"].startswith("frozen_synthetic")
