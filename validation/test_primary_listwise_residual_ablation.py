import json
from pathlib import Path

import numpy as np

from experiments.ablate_step3_primary_listwise_residual import (
    paired_statistics,
    summarize_rows,
)


def test_ablation_contract_uses_fresh_cohort_and_bounded_cpu():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "config" / "step3_primary_listwise_residual_ablation_v1.json").read_text()
    )
    assert contract["status"] == "frozen_before_primary_listwise_residual_ablation"
    assert not set(contract["ablation_seeds"]).intersection(contract["all_prior_seeds"])
    assert contract["arms"] == [
        "analytic_teacher", "learned_listwise_residual", "no_upper_band_removal"
    ]
    assert contract["workers"] * contract["threads_per_worker"] == 15
    assert contract["statistics"]["paired_bootstrap_resamples"] == 10000


def test_paired_statistics_preserve_pairing_and_effect_direction():
    contract = {
        "statistics": {
            "random_seed": 7,
            "paired_bootstrap_resamples": 1000,
            "confidence_level": 0.95,
            "wilcoxon_alternative": "two-sided",
        }
    }
    def rows(offset):
        return [{
            "seed": 1, "target_rank": rank,
            "delivery_ratio": 0.2 + offset,
            "stale_ratio": 0.1,
            "fnd_free_steps": 100 + rank,
            "episode_service_fairness": 0.8,
            "global_packets_per_j": 200.0,
        } for rank in range(3)]
    result = paired_statistics({"rows": rows(0.0)}, {"rows": rows(0.1)}, contract)
    delivery = result["delivery_ratio"]
    assert np.isclose(
        delivery["mean_paired_difference_treatment_minus_teacher"], 0.1
    )
    assert delivery["wins_ties_losses"] == {"wins": 3, "ties": 0, "losses": 0}
    assert result["fnd_free_steps"]["wilcoxon_two_sided_p_value"] == 1.0


def test_summarize_rows_uses_episode_fairness_and_energy_efficiency():
    rows = [{
        "joint_qos_pass": True,
        "delivery_ratio": 0.25,
        "stale_ratio": 0.05,
        "episode_service_fairness": 0.9,
        "fnd_free_steps": 120,
        "global_packets_per_j": 220.0,
    }]
    result = summarize_rows(rows, {"steps": 1})
    assert result["joint_qos_pass_count"] == 1
    assert np.isclose(result["mean_episode_service_fairness"], 0.9)
    assert result["mean_global_packets_per_j"] == 220.0
