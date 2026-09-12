from pathlib import Path

from experiments.run_v2_phase_b4_constraint_boundary_calibration import expand_candidates, load_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "v2_phase_b4_constraint_boundary_calibration_20260904.json"


def test_b4_preserves_acceptance_gate_and_seals_new_cohort():
    contract = load_contract(CONTRACT)
    assert contract["development_seeds"] == list(range(12100, 12110))
    assert contract["reserved_gate_seeds"] == list(range(12600, 12620))
    assert contract["gate_targets_unchanged"] == {
        "delivery_min": .45, "packets_per_j_min": 244.8, "stale_ratio_max": .035,
        "fairness_min": .92, "rmst_min": 128., "feasibility_violations_max": 0,
    }
    candidates = expand_candidates(contract)
    assert len(candidates) == 16
    assert all(row["fairness_guard"] < contract["gate_targets_unchanged"]["fairness_min"] for row in candidates)
    assert not set(range(3900, 3920)) & set(contract["development_seeds"])
