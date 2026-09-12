from pathlib import Path

import pytest

from experiments.run_v2_phase_a2_causal_accounting_repair import (
    analyze_gate,
    factorial_metric,
    load_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "v2_phase_a2_causal_accounting_repair_20260901.json"


def test_a2_contract_preserves_failed_a1_and_opened_seed_scope():
    contract, phase_a = load_contract(CONTRACT)
    assert contract["gate_a"]["threshold"] == 0.8
    assert contract["development_seeds"] == list(range(12000, 12020))
    assert not set(contract["development_seeds"]) & set(range(3900, 3920))
    assert phase_a["gate_a"]["gate_a_pass"] is False


def test_factorial_shapley_reconstructs_gap_exactly():
    aggregates = {
        "hta_count_hta_selection": {"mean_delivery_ratio": 1.0},
        "hta_count_residual_selection": {"mean_delivery_ratio": 1.2},
        "full_count_hta_selection": {"mean_delivery_ratio": 1.3},
        "full_count_residual_selection": {"mean_delivery_ratio": 1.5},
    }
    phase_a = {"aggregates": {
        "hta_mac_v1": {"mean_delivery_ratio": 1.0},
        "residual_energy_cap_corrected": {"mean_delivery_ratio": 1.5},
    }}
    result = factorial_metric(aggregates, phase_a, "delivery_ratio", 1e-12)
    assert result["count_rule_shapley"] + result["selection_rule_shapley"] == pytest.approx(0.5)
    assert result["unexplained_accounting_residual"] == pytest.approx(0.0)
    assert result["attribution_fraction"] == pytest.approx(1.0)


def test_unexplained_residual_reduces_attribution_below_gate():
    aggregates = {
        "hta_count_hta_selection": {"mean_delivery_ratio": 1.0},
        "hta_count_residual_selection": {"mean_delivery_ratio": 1.1},
        "full_count_hta_selection": {"mean_delivery_ratio": 1.1},
        "full_count_residual_selection": {"mean_delivery_ratio": 1.3},
    }
    phase_a = {"aggregates": {
        "hta_mac_v1": {"mean_delivery_ratio": 1.0},
        "residual_energy_cap_corrected": {"mean_delivery_ratio": 1.5},
    }}
    result = factorial_metric(aggregates, phase_a, "delivery_ratio", 1e-12)
    assert result["unexplained_accounting_residual"] == pytest.approx(0.2)
    assert result["attribution_fraction"] == pytest.approx(0.6)


def test_gate_requires_integrity_and_eighty_percent_for_both_metrics():
    aggregates = {
        arm: {
            "mean_delivery_ratio": delivery,
            "mean_packets_per_j": efficiency,
        }
        for arm, delivery, efficiency in (
            ("hta_count_hta_selection", 1.0, 10.0),
            ("hta_count_residual_selection", 1.1, 11.0),
            ("full_count_hta_selection", 1.2, 12.0),
            ("full_count_residual_selection", 1.5, 15.0),
        )
    }
    phase_a = {"aggregates": {
        "hta_mac_v1": {"mean_delivery_ratio": 1.0, "mean_packets_per_j": 10.0},
        "residual_energy_cap_corrected": {"mean_delivery_ratio": 1.5, "mean_packets_per_j": 15.0},
    }}
    contract, _ = load_contract(CONTRACT)
    passed = analyze_gate(aggregates, phase_a, contract, {"integrity": True})
    failed = analyze_gate(aggregates, phase_a, contract, {"integrity": False})
    assert passed["gate_a_pass"] is True
    assert failed["gate_a_pass"] is False
