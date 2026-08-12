import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from experiments.sweep_step3_primary_listwise_residual import (
    SetRemovalRanker,
    ranking_features,
    removal_count,
    teacher_winner,
)


def test_set_removal_ranker_is_permutation_equivariant():
    torch.manual_seed(3)
    model = SetRemovalRanker(5, 16)
    features = torch.randn(2, 7, 5)
    permutation = torch.tensor([4, 0, 6, 2, 1, 5, 3])
    direct = model(features)
    permuted = model(features[:, permutation])
    assert torch.allclose(permuted, direct[:, permutation], atol=1e-7, rtol=0.0)


def test_service_aware_features_expose_teacher_primary_order():
    env = SimpleNamespace(cumulative_service=np.asarray([2.0, 9.0, 5.0]))
    q_values = np.asarray([[0.0, 2.0, 3.0, 4.0]] * 3)
    action = np.asarray([1, 1, 1])
    caps = np.asarray([3, 3, 3])
    mask = np.asarray([True, True, True])
    features, eligible = ranking_features(env, q_values, action, caps, mask)
    assert eligible.all()
    assert int(np.argmax(features[:, 0])) == 1
    assert teacher_winner(env, q_values, action, mask) == 1


def test_removal_count_respects_lower_and_upper_qos_band():
    env = SimpleNamespace(
        step3_qos_counts={"demand": 100, "delivered": 20},
        members=np.asarray([0, 1]),
        base=SimpleNamespace(queue=np.asarray([10, 10]), alive=np.asarray([True, True])),
    )
    band = {"lower_delivery_target": 0.235, "upper_delivery_target": 0.300}
    assert removal_count(np.asarray([10, 10]), env, band) == 4


def test_listwise_contract_preserves_seed_firewall_and_cpu_allocation():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "config" / "step3_primary_listwise_residual_sweep_v1.json").read_text())
    assert contract["status"] == "frozen_before_primary_listwise_residual_sweep"
    assert not set(contract["development_seeds"]).intersection(contract["prohibited_seeds"])
    assert set(range(3400, 3405)).issubset(contract["prohibited_seeds"])
    assert contract["parallel_candidates"] * contract["threads_per_candidate"] == 16


def test_continuation_contract_does_not_relax_v1_gates():
    root = Path(__file__).resolve().parents[1]
    v1 = json.loads((root / "config" / "step3_primary_listwise_residual_sweep_v1.json").read_text())
    v2 = json.loads((root / "config" / "step3_primary_listwise_residual_continuation_v2.json").read_text())
    assert v2["status"] == "frozen_before_primary_listwise_residual_continuation"
    assert v2["gates"] == v1["gates"]
    assert v2["development_seeds"] == v1["development_seeds"]
    assert v2["prohibited_seeds"] == v1["prohibited_seeds"]


def test_confirmation_cohort_is_new_and_statistics_are_frozen():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads((root / "config" / "step3_primary_listwise_residual_confirmation_v1.json").read_text())
    assert contract["status"] == "frozen_before_listwise_residual_confirmation"
    assert not set(contract["confirmation_seeds"]).intersection(contract["development_and_prior_seeds"])
    assert contract["statistics"]["paired_bootstrap_resamples"] == 10000
    assert contract["statistics"]["confidence_level"] == 0.95
    assert contract["workers"] * contract["threads_per_worker"] == 16
