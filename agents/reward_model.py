"""Logged, scale-normalized HTA-MAC reward decomposition."""

from __future__ import annotations

from dataclasses import dataclass


TERM_ORDER = (
    "packets_delivered",
    "idle_energy_j",
    "deaths",
    "high_harvest_alignment",
    "declining_allocation",
    "queue_fairness",
)


@dataclass(frozen=True)
class RewardModel:
    scales: dict[str, float]
    weights: dict[str, float]

    def evaluate(self, raw_terms: dict[str, float]):
        weighted = {}
        for name in TERM_ORDER:
            scale = max(float(self.scales[name]), 1e-12)
            weighted[name] = (
                float(self.weights[name]) * float(raw_terms[name]) / scale
            )
        return float(sum(weighted.values())), weighted
