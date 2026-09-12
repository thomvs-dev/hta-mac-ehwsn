import json
from pathlib import Path

import pytest

from experiments.run_v2_phase_a_mechanism_audit import (
    gate_a_attribution,
    shapley_efficiency_decomposition,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "v2_phase_a_marginal_service_energy_audit_20260901.json"
RESULTS = ROOT / "outputs" / "phaseA" / "v2_marginal_service_energy_audit_20260901" / "results.json"


def _aggregate(delivery, packets, energy, under_service):
    return {
        "mean_delivery_ratio": delivery,
        "mean_global_packets": packets,
        "mean_network_energy_j": energy,
        "mean_under_service_capacity_ratio": under_service,
        "max_role_reconstruction_error_j": 0.0,
    }


def test_contract_is_phase_a_frozen_and_saved_pre_run_inventory_was_fresh():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    inventory = results["seed_inventory"]
    assert contract["phase"] == "A_only_no_training"
    assert contract["attribution_contract"]["threshold"] == 0.8
    assert not set(contract["development_seeds"]) & set(range(3900, 3920))
    assert inventory["development_seeds"] == contract["development_seeds"]
    assert inventory["development_overlap_registry"] == []
    assert inventory["development_overlap_opened_3900_3919"] == []
    assert inventory["fresh_development_cohort"] is True


def test_efficiency_shapley_reconstructs_exact_gap():
    result = shapley_efficiency_decomposition(100.0, 2.0, 120.0, 2.2)
    assert result["service_component_packets_per_j"] + result["energy_component_packets_per_j"] == pytest.approx(
        result["observed_gap_packets_per_j"], abs=1e-12
    )
    assert result["reconstruction_error"] == pytest.approx(0.0, abs=1e-12)


def test_gate_a_enforces_eighty_percent_exactly():
    aggregates = {
        "hta_mac_v1": _aggregate(0.40, 100.0, 1.0, 0.08),
        "residual_energy_cap_corrected": _aggregate(0.45, 105.0, 1.0, 0.04),
    }
    result = gate_a_attribution(aggregates, 0.8)
    assert result["delivery"]["attribution_fraction"] == pytest.approx(0.8)
    assert result["minimum_attribution_fraction"] == pytest.approx(0.8)
    assert result["gate_a_pass"] is True


def test_gate_a_does_not_weaken_failure():
    aggregates = {
        "hta_mac_v1": _aggregate(0.40, 100.0, 1.0, 0.07),
        "residual_energy_cap_corrected": _aggregate(0.45, 105.0, 1.0, 0.04),
    }
    result = gate_a_attribution(aggregates, 0.8)
    assert result["delivery"]["attribution_fraction"] == pytest.approx(0.6)
    assert result["gate_a_pass"] is False
    assert result["continuation_authorized"] is False
