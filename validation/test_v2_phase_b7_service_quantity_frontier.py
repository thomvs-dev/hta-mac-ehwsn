import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config/v2_phase_b7_service_quantity_frontier_20260904.json"


def test_b7_contract_freezes_exact_targets_and_new_cohorts():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "frozen_before_v2_phase_b7_service_quantity_frontier"
    assert contract["service_cap_candidates"] == [20, 21, 22, 23, 24]
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
    unavailable = {
        seed
        for first, last in contract["unavailable_seed_ranges"]
        for seed in range(first, last + 1)
    }
    assert len(development) == 10 and len(reserved) == 20
    assert development.isdisjoint(reserved | unavailable)
    assert reserved.isdisjoint(unavailable)
    assert set(range(3900, 3920)) <= unavailable
    assert contract["decision_rules"]["no_neural_training"] is True
