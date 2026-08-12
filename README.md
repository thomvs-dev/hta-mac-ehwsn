# HTA-MAC

HTA-MAC is a bounded intra-cluster MAC research framework for energy-harvesting
wireless sensor networks. It combines a frozen HEART-CH cluster-head schedule,
a permutation-equivariant branching C51 controller, explicit QoS accounting,
and a compact learned residual ranker for globally coupled slot removal.

Routing changes and cluster-head retraining are deliberately out of scope.

## Current verified result: final matched evaluation

The final preregistered evaluation used 20 previously unused independent seeds
(3700--3719), five target-rank schedules nested per seed, a 3,000-round
idle-listening-on horizon, and five matched policies. Statistical tests operate
on seed means (n=20), not the 100 correlated seed/rank rows.

| Policy | Joint QoS | Delivery | Stale | Fairness | FND | Packets/J |
|---|---:|---:|---:|---:|---:|---:|
| HTA-MAC learned residual | **99/100** | 0.279481 | 0.056102 | 0.858371 | 125.10 | 224.8928 |
| Analytic teacher | **99/100** | 0.279521 | 0.056109 | 0.858429 | 125.10 | 224.9213 |
| Tuned energy-proportional | 97/100 | 0.285246 | 0.060788 | 0.807922 | 132.43 | 225.5582 |
| S2A2MAC-adapted | 0/100 | 0.136319 | 0.155080 | 0.669100 | 140.53 | 231.8784 |
| FFSS-adapted | 1/100 | 0.208664 | 0.045595 | 0.959557 | 133.99 | 220.4545 |

Against tuned energy-proportional, HTA-MAC improved fairness by 0.05045 and
reduced stale ratio by 0.00469, but reduced delivery by 0.00576, reached FND
7.33 rounds earlier, and delivered 0.665 fewer packets/J. All five paired
differences remained significant after Holm correction across 15 prespecified
hypotheses. The supported contribution is reliable joint-QoS allocation with
an explicit QoS--lifetime trade-off, not universal lifetime superiority.

See
[`HTA_MAC_FINAL_MATCHED_BASELINE_REPORT_20260813.md`](HTA_MAC_FINAL_MATCHED_BASELINE_REPORT_20260813.md)
for confidence intervals, adjusted p-values, effect sizes, limitations, and
paper-safe claims.

## Residual confirmation

The primary idle-on track now includes a service-aware, permutation-equivariant
listwise residual ranker. It was selected on development seeds 2400--2404 and
confirmed on a separately frozen cohort, seeds 3500--3504.

| Confirmation metric (25 seed/rank pairs) | Analytic teacher | Learned residual |
|---|---:|---:|
| Joint QoS | 24/25 | 24/25 |
| Mean FND-free steps | 124.20 | 124.20 |
| Mean episode service fairness | 0.837725 | 0.837657 |
| Mean packets/J | 223.8858 | 223.8713 |
| Teacher changes / residual disagreement | 3,089 | 118 |

The residual ranker reduced action disagreement with the analytic teacher by
**96.18%** and passed every preregistered confirmation gate. Paired 95%
bootstrap intervals included zero for FND, fairness, and packets/J; two-sided
Wilcoxon p-values were 1.000, 0.817, and 0.089 respectively.

The supported claim is operational compatibility with the analytic teacher
within the frozen tolerances. This is a hybrid constrained controller: the
QoS-band removal count remains analytic. The evidence does not establish
teacher superiority, a fully unconstrained neural policy, or universal 25/25
QoS feasibility.

See
[`HTA_MAC_LISTWISE_RESIDUAL_CONFIRMATION_REPORT_20260812.md`](HTA_MAC_LISTWISE_RESIDUAL_CONFIRMATION_REPORT_20260812.md)
for the complete result, confidence intervals, limitations, and paper-safe
claim boundary.

## Fresh-cohort ablation

A separately preregistered ablation on seeds 3600--3604 compared analytic
ranking, learned ranking, and disabling upper-band removal:

| Arm | QoS | Delivery | FND-free | Fairness | Packets/J |
|---|---:|---:|---:|---:|---:|
| Analytic teacher | 25/25 | 0.278071 | 123.92 | 0.858214 | 225.7291 |
| Learned listwise residual | 25/25 | 0.277843 | 123.88 | 0.859354 | 225.7709 |
| No upper-band removal | 25/25 | 0.308043 | 121.84 | 0.831995 | 225.3260 |

The learned ranker again reduced teacher-action disagreement by **95.36%** and
passed every compatibility gate. Disabling upper-band removal increased mean
delivery by 0.02997, but advanced FND by 2.08 rounds, reduced fairness by
0.02622, and reduced packets/J by 0.40314. Paired 95% confidence intervals for
those four effects excluded zero.

The residual ranker has 225 parameters, adding 0.194% to the 116,033-parameter
base controller. Component-level median latency was similar to analytic ranking;
the learned p95 was higher, so no end-to-end speedup is claimed.

See
[`HTA_MAC_LISTWISE_RESIDUAL_ABLATION_REPORT_20260813.md`](HTA_MAC_LISTWISE_RESIDUAL_ABLATION_REPORT_20260813.md)
for paired effect sizes, confidence intervals, p-values, profiling methodology,
and limitations.

## Method

The learned residual module addresses a failure of earlier full-action and
branch-local distillation objectives. Those objectives did not observe the
cumulative per-node service used by the teacher and could not reliably reproduce
a globally budget-coupled removal order.

The replacement uses:

- a shared node scorer, so score permutations follow node permutations;
- cumulative-service percentile and fraction features;
- standardized marginal Q-loss and action-cap features;
- listwise cross-entropy over the teacher's next removal;
- one DAgger-style on-policy aggregation cycle;
- deterministic QoS-band and global-budget projection.

The selected residual checkpoint contains only 3.7 KB of serialized model and
metadata. Its SHA-256 is:

```text
9df103c19d1f80336f80514d54bb88ee43cda31ca9749214160404e77a625c57
```

## Repository layout

```text
agents/       branching C51, QoS, and CH-risk components
envs/         identity-safe scheduled MAC environments and QoS accounting
experiments/  training, listwise selection, and confirmation entry points
config/       frozen experiment, seed, threshold, and statistics contracts
validation/   unit, invariance, accounting, and contract tests
outputs/      selected checkpoints and compact machine-readable evidence
```

## Reproduce the verified checks

From the repository root:

```powershell
python -B -m pytest validation -q -p no:cacheprovider
```

Expected result for this revision:

```text
139 passed
```

Rerun the frozen independent confirmation:

```powershell
$env:OMP_NUM_THREADS='8'
$env:MKL_NUM_THREADS='8'
$env:OPENBLAS_NUM_THREADS='8'
$env:NUMEXPR_NUM_THREADS='8'

python -B -m experiments.confirm_step3_primary_listwise_residual `
  --contract config/step3_primary_listwise_residual_confirmation_v1.json `
  --output outputs/phase3/step3_primary_listwise_residual_confirmation_v1/summary.json
```

The confirmation runner uses two eight-thread processes. It verifies every
checkpoint/config hash before opening the frozen cohort and returns a nonzero
exit code if any operational gate fails.

## Authoritative artifacts

- Selected base controller:
  `outputs/phase2/step3_primary_idle_hybrid_100ep_opt6801_cpu6/branching_c51.pt`
- Selected residual ranker:
  `outputs/phase2/step3_primary_listwise_residual_continuation_v2/continue_lr3e3_ep8/removal_ranker.pt`
- Development selection:
  `outputs/phase2/step3_primary_listwise_residual_continuation_v2/summary.json`
- Independent confirmation:
  `outputs/phase3/step3_primary_listwise_residual_confirmation_v1/summary.json`
- Fresh-cohort ablation:
  `outputs/phase3/step3_primary_listwise_residual_ablation_v1/summary.json`
- Final matched-baseline evaluation:
  `outputs/phase3/step3_final_matched_baseline_evaluation_v1/summary.json`
- Publication tables and figures:
  `outputs/phase3/step3_final_matched_baseline_evaluation_v1/publication/`
- Frozen confirmation contract:
  `config/step3_primary_listwise_residual_confirmation_v1.json`
- Frozen final-evaluation contract:
  `config/step3_final_matched_baseline_evaluation_v1.json`

Historical Phase 0--3 status files remain available for provenance:

- `PHASE0_STATUS.md`
- `PHASE1_STATUS.md`
- `PHASE2_STATUS.md`
- `PHASE3_STATUS.md`
- `BASELINE_PROVENANCE.md`
- `PRE_PHASE2_DECISION_CLOSURE.md`

## Non-negotiable scope

- frozen HEART-CH cluster-head selection and shared exogenous schedules;
- no cluster-head retraining and no routing modification;
- identity-preserving, schedule-matched training and evaluation;
- explicit cohort-consistent delivery, stale-drop, and service-fairness metrics;
- paired comparisons and frozen thresholds before confirmation;
- no cross-paper numerical superiority claim under incompatible simulators.
