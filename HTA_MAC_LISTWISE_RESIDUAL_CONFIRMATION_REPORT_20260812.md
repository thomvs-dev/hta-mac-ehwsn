# HTA-MAC Listwise Residual Ranker: Selection and Independent Confirmation

**Date:** 12 August 2026
**Scope:** frozen HEART-CH, MAC-only primary idle-on track
**Outcome:** development selection passed and independent confirmation passed

## Executive result

A permutation-equivariant listwise residual ranker was implemented to learn the
globally coupled slot-removal priority that the earlier branch-wise objectives
could not represent. The selected model passed all frozen development gates and
then passed all frozen operational gates on a new confirmation cohort.

The paper-safe result is:

> On five new schedule seeds and 25 seed/rank pairs, the learned listwise
> residual ranker reduced action disagreement with the frozen analytic teacher
> by 96.18% relative to the teacher's 3,089 changed slots, while matching its
> 24/25 joint-QoS count and showing no detected paired difference in FND,
> fairness, or packets/J at the predeclared 5% significance level.

This is evidence for a compact **hybrid learned residual controller**, not a
claim that an unconstrained neural policy independently discovered the complete
QoS rule or outperformed the analytic teacher.

## Why the earlier objectives failed

The analytic teacher removes slots lexicographically using cumulative per-node
service and marginal Q loss. Cumulative service was not present in the base
controller observation. Full-action and local branch imitation were therefore
asked to infer a decision from incomplete information, and local changes also
altered the globally projected action ranking.

The repair added only permutation-safe residual inputs:

- active-set cumulative-service percentile and fraction;
- standardized negative marginal Q loss;
- current action and feasible cap fractions.

A shared node scorer produces an equivariant removal ranking. The global QoS
band determines the number of residual removals. One DAgger-style on-policy
aggregation round exposes the ranker to states induced by its own decisions.

This design follows three established ideas:

- policy distillation from a stronger teacher ([Rusu et al., 2015](https://arxiv.org/abs/1511.06295));
- permutation-aware set ranking with cross-item context ([Pang et al., 2019](https://arxiv.org/abs/1912.05891));
- dataset aggregation to reduce sequential imitation distribution shift
  ([Ross et al., 2011](https://proceedings.mlr.press/v15/ross11a.html)).

Listwise likelihood is also consistent with established ranking formulations
such as [SQL-Rank](https://proceedings.mlr.press/v80/wu18c.html). These sources
motivate the method; they do not validate this repository's measured results.

## Development selection

Four initial candidates were evaluated on development seeds 2400--2404. The
best initial candidate passed every operational gate but reached only 84.67%
validation top-1 accuracy against a frozen 90% requirement. The threshold was
not relaxed post hoc.

A separately frozen continuation sweep retained the same gates. The selected
candidate was `continue_lr3e3_ep8`:

| Metric | Frozen requirement | Selected result |
|---|---:|---:|
| Validation top-1 accuracy | >= 90% | 91.11% |
| Teacher-correction reduction | >= 90% | 96.72% |
| Joint QoS | >= 23/25 | 25/25 |
| Mean FND-free steps | baseline - 5 | 125.28 vs. 125.24 |
| Mean fairness | baseline - 0.01 | 0.8606 vs. 0.8601 |
| Mean packets/J | >= 99% baseline | 225.25 vs. 225.25 |

The selected checkpoint was frozen before confirmation:

`9df103c19d1f80336f80514d54bb88ee43cda31ca9749214160404e77a625c57`

## Independent confirmation

Confirmation seeds 3500--3504 were not part of development or the earlier
3100/3400 cohorts. The teacher and residual arms were executed on the same 25
seed/rank pairs using two concurrent eight-thread workers.

| Metric | Analytic teacher | Learned residual |
|---|---:|---:|
| Joint QoS | 24/25 | 24/25 |
| Mean FND-free steps | 124.20 | 124.20 |
| Mean fairness | 0.837725 | 0.837657 |
| Mean packets/J | 223.8858 | 223.8713 |
| Changed/disagreement slots | 3,089 teacher changes | 118 residual-teacher disagreement |
| Relative disagreement reduction | -- | 96.18% |

The single failed QoS pair was identical in both arms: seed 3501, rank 0,
delivery 0.2061, stale ratio 0.0954, fairness 0.6998. The learned ranker did not
introduce a new failed pair, but this shared failure prevents a 25/25 global-QoS
claim.

## Paired uncertainty analysis

Confidence intervals use 10,000 paired bootstrap resamples. P-values use a
two-sided paired Wilcoxon test.

| Metric (residual - teacher) | Mean difference | 95% paired-bootstrap CI | Wilcoxon p |
|---|---:|---:|---:|
| FND-free steps | 0.0000 | [0.0000, 0.0000] | 1.0000 |
| Fairness | -0.0000685 | [-0.0005502, 0.0004049] | 0.8173 |
| Packets/J | -0.014509 | [-0.030577, 0.000038] | 0.0894 |

All intervals include zero and all p-values exceed 0.05. This supports
compatibility/equivalence within the tested tolerances; it is not evidence that
the learned ranker is superior to the teacher.

## Frozen gates and integrity

- All five confirmation gates passed: QoS, correction reduction, FND,
  fairness, and packets/J.
- Confirmation thresholds were frozen before seeds 3500--3504 were opened.
- The confirmation report records `confirmation_seeds_opened: true`.
- Final repository validation: 132 passed.
- Four development candidates used 16 CPU threads; the confirmation used two
  eight-thread workers.

## Contribution and limitation boundary

Supported:

- a permutation-equivariant, service-aware listwise residual ranker can compress
  most of the analytic teacher's node-selection behavior;
- the result generalized to a separate confirmation cohort within frozen
  operational tolerances;
- paired uncertainty estimates show no detected degradation in FND, fairness,
  or packets/J relative to the teacher.

Not supported:

- superiority over the analytic teacher;
- a fully shield-free or purely learned controller—the QoS-band removal count
  remains an analytic constraint component;
- universal 25/25 QoS feasibility;
- broad cross-paper superiority under incompatible simulators.

## Next paper sequence

1. Treat the hybrid residual ranker as the primary implementation contribution.
2. Add an ablation: analytic teacher ranking versus learned listwise ranking
   versus no upper-band removal, keeping the removal count fixed where relevant.
3. Report paired effect sizes and confidence intervals, not only p-values.
4. Profile inference time and parameter count to support the compression claim.
5. Keep the identical seed/rank failure visible and analyze its demand/energy
   mechanism instead of hiding it.
6. Compare with literature only under explicitly matched environment assumptions.

## Artifacts

- Development contract: `config/step3_primary_listwise_residual_sweep_v1.json`
- Continuation contract: `config/step3_primary_listwise_residual_continuation_v2.json`
- Confirmation contract: `config/step3_primary_listwise_residual_confirmation_v1.json`
- Ranker and sweep: `experiments/sweep_step3_primary_listwise_residual.py`
- Confirmation evaluator: `experiments/confirm_step3_primary_listwise_residual.py`
- Selected checkpoint:
  `outputs/phase2/step3_primary_listwise_residual_continuation_v2/continue_lr3e3_ep8/removal_ranker.pt`
- Development selection:
  `outputs/phase2/step3_primary_listwise_residual_continuation_v2/summary.json`
- Independent confirmation:
  `outputs/phase3/step3_primary_listwise_residual_confirmation_v1/summary.json`
