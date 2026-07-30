"""Tests for preregistered paired nonparametric statistics."""

import numpy as np

from core.paired_statistics import (
    hodges_lehmann_paired_difference,
    matched_pairs_rank_biserial,
    paired_wilcoxon_effect,
)


def test_paired_effects_have_declared_orientation():
    first = np.array([4.0, 5.0, 6.0])
    second = np.array([1.0, 2.0, 3.0])
    result = paired_wilcoxon_effect(first, second)
    assert result["median_paired_difference"] == 3.0
    assert result["hodges_lehmann_paired_difference"] == 3.0
    assert result["matched_pairs_rank_biserial"] == 1.0
    assert result["first_wins"] == 3


def test_paired_effects_handle_all_ties():
    values = np.array([1.0, 2.0, 3.0])
    result = paired_wilcoxon_effect(values, values)
    assert result["statistic"] is None
    assert result["p_value_two_sided"] == 1.0
    assert result["matched_pairs_rank_biserial"] == 0.0


def test_hodges_lehmann_uses_walsh_averages_not_only_raw_median():
    first = np.array([0.0, 0.0, 10.0, 10.0])
    second = np.zeros(4)
    assert hodges_lehmann_paired_difference(first, second) == 5.0
    assert matched_pairs_rank_biserial(first, second) == 1.0