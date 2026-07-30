# Phase 3 baseline provenance and adaptation limits

This document records what was verified from the primary papers before the
Phase 3 implementations were written. It supersedes the earlier description of
S2A2MAC as a purely cluster-wide HMM threshold.

## S2A2MAC

Primary source: P. Movva, K. K. Kamarajugadda, and P. T. R. Polipalli,
"An energy aware cluster-based routing and adaptive semi-synchronized MAC for
energy harvesting WSN," *International Journal of Communication Systems*,
vol. 35, no. 12, e5202, 2022. DOI:
https://doi.org/10.1002/dac.5202.

Verified structure:

- odd and even clusters alternate between sleep and active periods;
- nodes within a cluster share the cluster sleep period;
- an HMM categorizes individual nodes using residual energy and current load;
- three node categories receive one, two, or three active mini-slots.

Therefore, the earlier claim that S2A2MAC structurally cannot differentiate
nodes within a cluster is false and must not appear in the manuscript.

The paper does not provide a reusable HMM checkpoint or complete numerical
transition/emission parameter artifact for direct transplantation into this
simulator. `S2A2MACAdaptedPolicy` consequently preserves alternating cluster
sleep and the three residual-energy/load node layers, using deterministic
cluster-local tertiles to make the missing categorization parameters explicit
and reproducible. It is an adaptation, not a bit-exact reproduction.

## FFSS

Primary source: S. Gong, X. Liu, K. Zheng, W. Lu, and Y.-H. Zhu, "TDMA
scheduling schemes targeting high channel utilization for energy-harvesting
wireless sensor networks," *IET Communications*, vol. 15, pp. 2097-2110,
2021. DOI: https://doi.org/10.1049/cmu2.12243.

Verified structure:

- FFSS uses a fixed frame with the number of slots equal to the number of nodes;
- each node receives exactly one slot for fairness;
- upcoming energy and data enter the slot-assignment objective;
- a Hungarian-based assignment places immediately feasible nodes toward front
  slots and nodes needing time to acquire energy/data toward later slots.

The current HTA-MAC environment operates at whole-round resolution and exposes
slot counts, not within-frame slot order or within-frame energy/data arrivals.
An exact FFSS ordering experiment is therefore impossible without changing the
common environment. `FFSSAdaptedPolicy` preserves fixed-frame, one-slot-per-node,
feasible-first selection when a cluster exceeds `T=24`; it must be labelled
"FFSS-adapted," never an exact FFSS reproduction.

## Seven-policy resolution

The six named comparison policies are static equal, energy proportional,
harvest proportional, S2A2MAC-adapted, FFSS-adapted, and HTA-MAC. A seeded
random-budgeted allocator is retained as the seventh diagnostic policy. It is
not presented as a literature baseline.
