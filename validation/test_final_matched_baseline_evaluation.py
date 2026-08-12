import json
from pathlib import Path

import numpy as np

from experiments.evaluate_step3_final_matched_baselines import (
    holm_adjust,
    load_contract,
    seed_means,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "step3_final_matched_baseline_evaluation_v1.json"


def test_final_contract_is_frozen_disjoint_and_seed_is_the_independent_unit():
    contract = load_contract(CONTRACT)
    assert contract["independent_unit"] == "seed"
    assert len(contract["evaluation_seeds"]) == 20
    assert set(contract["evaluation_seeds"]).isdisjoint(contract["all_prior_seeds"])
    assert contract["selection_or_retuning_permitted"] is False
    assert len(contract["inferential_comparators"]) * len(contract["statistics"]["metrics"]) == 15


def test_seed_means_collapse_nested_ranks_before_inference():
    rows = [
        {"seed": 1, "target_rank": 0, "delivery_ratio": 0.2, "joint_qos_pass": False},
        {"seed": 1, "target_rank": 1, "delivery_ratio": 0.4, "joint_qos_pass": True},
        {"seed": 2, "target_rank": 0, "delivery_ratio": 0.8, "joint_qos_pass": True},
    ]
    collapsed = seed_means(rows, {"delivery_ratio": "higher"})
    assert set(collapsed) == {1, 2}
    assert np.isclose(collapsed[1]["delivery_ratio"], 0.3)
    assert np.isclose(collapsed[1]["joint_qos_pass_rate"], 0.5)


def test_holm_adjustment_is_monotone_in_sorted_p_values():
    records = [
        {"wilcoxon_p_value_unadjusted": value}
        for value in (0.03, 0.001, 0.01, 0.2)
    ]
    holm_adjust(records)
    ordered = sorted(records, key=lambda row: row["wilcoxon_p_value_unadjusted"])
    adjusted = [row["wilcoxon_p_value_holm"] for row in ordered]
    assert adjusted == sorted(adjusted)
    assert np.allclose(adjusted, [0.004, 0.03, 0.06, 0.2])


def test_contract_json_declares_all_five_metric_directions():
    payload = json.loads(CONTRACT.read_text())
    assert payload["statistics"]["metrics"] == {
        "delivery_ratio": "higher",
        "stale_ratio": "lower",
        "episode_service_fairness": "higher",
        "fnd_free_steps": "higher",
        "global_packets_per_j": "higher",
    }
