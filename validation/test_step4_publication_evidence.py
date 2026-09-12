import json
from pathlib import Path

import numpy as np
import pytest

from experiments.evaluate_step4_publication_evidence import (
    aggregate,
    independent_schedule_bundle,
    load_contract,
)
from tools.fetch_pvgis_solar_trace import parse_hourly


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_step4_contract_keeps_development_and_confirmation_separate():
    contract = load_contract(ROOT / "config" / "step4_publication_evidence_development_v1.json")
    assert set(contract["development_seeds"]).isdisjoint(contract["prohibited_prior_seeds"])
    assert set(contract["development_seeds"]).isdisjoint(contract["confirmation_seeds"])
    assert contract["confirmation_seeds_opened"] is False
    assert len(contract["pareto_operating_points"]) == 3
    assert len(contract["transfer_scenarios"]) == 9
    assert contract["analysis_rules"]["development_is_descriptive_only"] is True


@pytest.mark.parametrize("nodes,expected_heads", [(20, 1), (50, 2), (100, 5)])
def test_independent_schedule_is_complete_and_dimensionally_valid(nodes, expected_heads):
    bundle = independent_schedule_bundle(
        seed=3800,
        nodes=nodes,
        horizon=12,
        field_size=100.0,
        bs_position=(50.0, 175.0),
        solar_states=3,
        thermal_states=3,
    )
    assert bundle["schedule_metadata"]["complete"] is True
    assert bundle["schedule_metadata"]["generator"] == "independent_balanced_rotation"
    assert len(bundle["schedule"]) == 12
    for frame in bundle["schedule"]:
        assert frame["positions"].shape == (nodes, 2)
        assert frame["stgcn_embedding"].shape[0] == nodes
        assert len(frame["cluster_heads"]) == expected_heads
        assert len(np.unique(frame["cluster_heads"])) == expected_heads


def test_aggregate_reports_censoring_instead_of_inventing_fnd():
    base = {
        "seed": 3800,
        "joint_qos_pass": True,
        "delivery_ratio": 0.25,
        "stale_ratio": 0.05,
        "episode_service_fairness": 0.85,
        "restricted_survival_rounds": 3000,
        "global_packets_delivered": 100,
        "network_energy_j": 1.0,
        "global_packets_per_j": 100.0,
        "allocated_slots": 20,
    }
    rows = [
        {**base, "fnd_event_observed": False},
        {**base, "seed": 3801, "fnd_event_observed": True, "restricted_survival_rounds": 120},
    ]
    result = aggregate(rows)
    assert result["fnd_event_count"] == 1
    assert result["fnd_censored_count"] == 1
    assert result["mean_restricted_survival_rounds"] == 1560.0


def test_pvgis_parser_requires_long_positive_trace():
    payload = {
        "outputs": {
            "hourly": [
                {"time": f"20200101:{index:04d}", "G(i)": float(index % 12)}
                for index in range(24 * 300)
            ]
        }
    }
    parsed = parse_hourly(payload)
    assert len(parsed) == 24 * 300
    assert max(value for _, value in parsed) == 11.0
    with pytest.raises(RuntimeError, match="hourly rows"):
        parse_hourly({"outputs": {"hourly": [{"time": "x", "G(i)": 1.0}]}})


def test_contract_rejects_open_confirmation_cohort():
    source = ROOT / "config" / "step4_publication_evidence_development_v1.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["confirmation_seeds_opened"] = True
    candidate = ROOT / "outputs" / "validation_artifacts" / "step4_invalid_open_confirmation.json"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    try:
        candidate.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RuntimeError, match="sealed"):
            load_contract(candidate)
    finally:
        candidate.unlink(missing_ok=True)
