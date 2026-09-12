from pathlib import Path

from experiments.run_v2_phase_b_teacher_gate import gate_checks, load_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "v2_phase_b_teacher_gate_20260901.json"


def test_gate_contract_freezes_selected_teacher_and_fresh_20_seed_cohort():
    contract, selected = load_contract(CONTRACT)
    assert selected["candidate"]["candidate_id"] == "mu_20_e4_s2_f3"
    assert contract["gate_seeds"] == list(range(12200, 12220))
    assert not set(contract["gate_seeds"]) & set(contract["calibration_seeds"])


def test_gate_b_requires_every_threshold_without_weakening():
    targets = {
        "delivery_min": .45, "packets_per_j_min": 244.8,
        "stale_ratio_max": .035, "fairness_min": .92,
        "rmst_min": 128, "feasibility_violations_max": 0,
    }
    exact = {
        "mean_delivery_ratio": .45, "mean_packets_per_j": 244.8,
        "mean_stale_ratio": .035, "mean_fairness": .92,
        "mean_rmst": 128, "feasibility_violations": 0,
    }
    assert all(gate_checks(exact, targets).values())
    for field, bad in (
        ("mean_delivery_ratio", .449999), ("mean_packets_per_j", 244.799),
        ("mean_stale_ratio", .035001), ("mean_fairness", .919999),
        ("mean_rmst", 127.999), ("feasibility_violations", 1),
    ):
        candidate = dict(exact); candidate[field] = bad
        assert not all(gate_checks(candidate, targets).values())
