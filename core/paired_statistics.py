"""Paired nonparametric statistics shared by Phase 3 and Phase 4 analyses."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, wilcoxon


def hodges_lehmann_paired_difference(first, second):
    """Return the paired Hodges-Lehmann shift (median Walsh average)."""
    differences = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("paired samples must be non-empty one-dimensional arrays")
    walsh = [
        (differences[i] + differences[j]) / 2.0
        for i in range(differences.size)
        for j in range(i, differences.size)
    ]
    return float(np.median(np.asarray(walsh, dtype=np.float64)))


def matched_pairs_rank_biserial(first, second):
    """Return signed matched-pairs rank-biserial correlation, first minus second."""
    differences = np.asarray(first, dtype=np.float64) - np.asarray(
        second, dtype=np.float64
    )
    nonzero = differences[differences != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    denominator = float(ranks.sum())
    positive = float(ranks[nonzero > 0.0].sum())
    negative = float(ranks[nonzero < 0.0].sum())
    return (positive - negative) / denominator


def paired_wilcoxon_effect(first, second):
    """Two-sided Wilcoxon test plus paired effect sizes, robust to all ties."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or first.size == 0:
        raise ValueError("paired samples must have the same non-empty 1-D shape")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("paired samples must be finite")
    differences = first - second
    if np.all(differences == 0.0):
        statistic = None
        p_value = 1.0
    else:
        test = wilcoxon(first, second, alternative="two-sided", method="auto")
        statistic = float(test.statistic)
        p_value = float(test.pvalue)
    return {
        "paired_trials": int(first.size),
        "statistic": statistic,
        "p_value_two_sided": p_value,
        "median_paired_difference": float(np.median(differences)),
        "hodges_lehmann_paired_difference": hodges_lehmann_paired_difference(
            first, second
        ),
        "matched_pairs_rank_biserial": matched_pairs_rank_biserial(
            first, second
        ),
        "first_wins": int(np.count_nonzero(differences > 0.0)),
        "ties": int(np.count_nonzero(differences == 0.0)),
        "first_losses": int(np.count_nonzero(differences < 0.0)),
    }