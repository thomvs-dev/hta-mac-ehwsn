# HTA-MAC Listwise Residual Ranker: Fresh-Cohort Ablation

**Date:** 13 August 2026
**Scope:** frozen HEART-CH, MAC-only primary idle-on track
**Outcome:** all preregistered integrity gates passed

## Question

This ablation separates two contributions that the confirmation experiment did
not isolate:

1. Can the learned listwise ranker continue to substitute for the analytic
   removal ranking on another unseen schedule cohort?
2. Does upper-band removal do useful work, or can it be omitted in favor of
   serving more packets?

Seeds 3600--3604 were frozen before evaluation and were disjoint from all
development, prior confirmation, and earlier reserved cohorts. Three arms ran
on the same 25 seed/rank pairs:

- analytic upper-band removal and analytic node ranking;
- the same removal-count rule with learned listwise node ranking;
- lower-floor service only, with upper-band removal disabled.

## Aggregate results

| Arm | QoS | Delivery | Stale | FND-free | Fairness | Packets/J |
|---|---:|---:|---:|---:|---:|---:|
| Analytic teacher | 25/25 | 0.278071 | 0.056961 | 123.92 | 0.858214 | 225.7291 |
| Learned listwise residual | 25/25 | 0.277843 | 0.057111 | 123.88 | 0.859354 | 225.7709 |
| No upper-band removal | 25/25 | 0.308043 | 0.055705 | 121.84 | 0.831995 | 225.3260 |

The learned ranker reduced teacher-action disagreement by 95.36% and passed
every frozen compatibility gate.

## Learned ranking versus analytic ranking

Paired differences are learned residual minus analytic teacher.

| Metric | Mean difference | 95% bootstrap CI | Wilcoxon p | Paired dz |
|---|---:|---:|---:|---:|
| Delivery | -0.000228 | [-0.000623, -0.000007] | 0.2695 | -0.245 |
| Stale ratio | +0.000150 | [-0.000042, +0.000475] | 0.8076 | +0.202 |
| FND-free steps | -0.04 | [-0.16, +0.08] | 0.7245 | -0.114 |
| Fairness | +0.001140 | [-0.000406, +0.003605] | 0.9138 | +0.206 |
| Packets/J | +0.041755 | [-0.015454, +0.140835] | 0.2339 | +0.179 |

The delivery bootstrap interval is narrowly below zero, but the absolute change
is 0.000228, the paired effect is small, the Wilcoxon test is not significant,
QoS remains 25/25, and all preregistered operational tolerances pass. This is
reported as a small measurable discrepancy, not exact identity.

## Removing the upper-band controller

Paired differences are no-upper-removal minus analytic teacher.

| Metric | Mean difference | 95% bootstrap CI | Wilcoxon p | Paired dz |
|---|---:|---:|---:|---:|
| Delivery | +0.029972 | [+0.020454, +0.040607] | 5.96e-8 | +1.130 |
| Stale ratio | -0.001256 | [-0.002587, +0.000027] | 0.1135 | -0.367 |
| FND-free steps | -2.08 | [-3.04, -1.20] | 0.000248 | -0.874 |
| Fairness | -0.026218 | [-0.036733, -0.016517] | 7.50e-5 | -0.996 |
| Packets/J | -0.403137 | [-0.648309, -0.179104] | 0.00342 | -0.676 |

Disabling removal increases delivery in every pair, but the extra service is
not free: FND occurs earlier in 15 pairs and is unchanged in 10; fairness falls
in 20 of 25 pairs; and energy efficiency falls on average. All three adverse
confidence intervals exclude zero. This supports upper-band removal as a
service-lifetime-fairness control, rather than an arbitrary post-processing
step.

The no-upper arm remained at 25/25 QoS because the frozen QoS contract specifies
a minimum delivery floor, stale ceiling, and fairness floor—not a penalty for
exceeding the learned service band. The ablation therefore demonstrates a
trade-off within feasible policies, not a feasibility rescue.

## Compactness and component latency

| Quantity | Result |
|---|---:|
| Residual-ranker parameters | 225 |
| Base C51-controller parameters | 116,033 |
| Parameter overhead | 0.194% |
| Ranker checkpoint | 3,747 bytes |
| Estimated ranker MACs/node | 192 |
| Estimated MACs at 100 nodes | 19,200 |
| Learned ranker median component latency | 120.30 us |
| Analytic ranking median component latency | 122.85 us |
| Learned ranker p95 component latency | 199.53 us |
| Analytic ranking p95 component latency | 173.52 us |

Latency uses 2,000 single-threaded iterations after 200 warmups. It measures one
removal-choice component only and excludes the environment, C51 forward pass,
QoS-count arithmetic, and repeated multi-removal projection. The median timings
are similar, while the learned p95 is higher; no end-to-end speedup claim is
supported.

## Paper-safe conclusions

Supported:

- the learned ranker generalizes across a second unseen five-seed cohort and
  remains within all operational compatibility tolerances;
- upper-band removal produces a statistically and practically visible trade-off:
  lower delivery but later FND, higher fairness, and higher packets/J;
- the learned residual adds only 225 parameters, or 0.194% over the base C51
  controller.

Not supported:

- exact numerical identity between learned and analytic ranking;
- learned-ranker superiority over the analytic teacher;
- end-to-end inference acceleration;
- cross-paper numerical superiority under unmatched simulators;
- a shield-free controller, because removal count remains analytic.

## Artifacts

- Frozen contract: `config/step3_primary_listwise_residual_ablation_v1.json`
- Runner: `experiments/ablate_step3_primary_listwise_residual.py`
- Validation: `validation/test_primary_listwise_residual_ablation.py`
- Machine-readable evidence:
  `outputs/phase3/step3_primary_listwise_residual_ablation_v1/summary.json`
