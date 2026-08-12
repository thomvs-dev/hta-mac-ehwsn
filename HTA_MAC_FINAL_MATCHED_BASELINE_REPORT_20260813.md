# HTA-MAC final matched-baseline evaluation

**Date:** 13 August 2026

**Status:** complete; all preregistered integrity gates passed

**Evidence:** `outputs/phase3/step3_final_matched_baseline_evaluation_v1/summary.json`

## Executive finding

The confirmed learned residual controller is a strong **joint-QoS controller**, not a universally superior lifetime controller.

On 20 previously unused independent seeds, with five matched target-rank schedules nested within each seed, HTA-MAC passed the joint delivery, stale-drop, and fairness constraints in **99/100** trials. Tuned energy-proportional passed 97/100, FFSS-adapted 1/100, and S2A2MAC-adapted 0/100.

The strongest honest comparison is against tuned energy-proportional. HTA-MAC reduced the stale ratio by 0.00469 and increased episode service fairness by 0.05045, but reduced delivery by 0.00576, reached first-node death 7.33 rounds earlier, and delivered 0.665 fewer packets/J. All five differences remained significant after Holm correction across the prespecified 15-test family.

This resolves an important ambiguity: the model's publication value is **reliable multi-constraint service allocation and a compact learned approximation of globally coupled analytic correction**, not a claim that it maximizes every energy and lifetime endpoint.

## Frozen evaluation design

- Final seeds: 3700-3719; none appeared in development, confirmation, or ablation evidence.
- Independent inferential unit: seed, n=20.
- Nested repeated subcases: five target-rank schedules per seed.
- Physical horizon: 3,000 rounds; every FND event was observed, so no FND value was right-censored.
- Environment: primary idle-listening-on scheduled MAC environment.
- Scope: frozen HEART-CH schedule, no routing changes, MAC actions only.
- Policies: learned listwise residual, analytic teacher, tuned energy-proportional, S2A2MAC-adapted, and FFSS-adapted.
- Inference: seed-level paired differences, two-sided Wilcoxon signed-rank tests, 20,000 paired-bootstrap resamples, 95% confidence intervals, and Holm family-wise correction over three comparators by five metrics.
- Post-cohort selection or tuning: prohibited and not performed.

Treating the seed as the independent unit is essential. The 100 seed/rank rows are correlated because five ranks share the same stochastic seed. Testing all 100 as independent observations would overstate precision.

## Aggregate results

| Policy | Joint QoS | Delivery | Stale | Fairness | FND | Packets/J |
|---|---:|---:|---:|---:|---:|---:|
| HTA-MAC learned residual | **99/100** | 0.279481 | 0.056102 | 0.858371 | 125.10 | 224.8928 |
| Analytic teacher | **99/100** | 0.279521 | 0.056109 | 0.858429 | 125.10 | 224.9213 |
| Energy-proportional, exponent 4 | 97/100 | **0.285246** | 0.060788 | 0.807922 | 132.43 | 225.5582 |
| S2A2MAC-adapted | 0/100 | 0.136319 | 0.155080 | 0.669100 | **140.53** | **231.8784** |
| FFSS-adapted | 1/100 | 0.208664 | **0.045595** | **0.959557** | 133.99 | 220.4545 |

Bold values are descriptive column maxima or minima where appropriate; they do not imply a single globally best policy. S2A2MAC's long FND and high packets/J arise alongside severe under-service and zero joint-QoS passes. FFSS achieves high fairness and low staleness while failing the delivery floor in nearly every trial.

## Paired HTA-MAC comparisons

All differences below are HTA-MAC minus comparator. The confidence intervals operate on independent seed-level means.

### Versus tuned energy-proportional

| Metric | Mean difference | 95% bootstrap CI | Holm p | Favorable to HTA? |
|---|---:|---:|---:|---|
| Delivery ratio | -0.005764 | [-0.009652, -0.002138] | 0.01923 | No |
| Stale ratio | -0.004686 | [-0.005976, -0.003328] | 0.000134 | Yes |
| Service fairness | +0.050449 | [+0.037771, +0.063212] | 0.000040 | Yes |
| FND rounds | -7.33 | [-8.36, -6.24] | 0.000353 | No |
| Packets/J | -0.6654 | [-0.8687, -0.4547] | 0.000114 | No |

The FND direction was consistent in all 20 seeds. The fairness gain occurred in 19/20 seeds; the stale-ratio improvement occurred in 18/20.

### Versus S2A2MAC-adapted

HTA-MAC improved delivery by 0.14316 and fairness by 0.18927, and reduced stale ratio by 0.09898. All three favorable directions occurred in 20/20 seeds and remained Holm-significant. The cost was 15.43 earlier FND rounds and 6.986 fewer packets/J. S2A2MAC therefore behaves as an energy-conservative under-service policy in this matched environment.

### Versus FFSS-adapted

HTA-MAC improved delivery by 0.07082 and packets/J by 4.4383 in all 20 seeds. FFSS produced 0.01051 lower staleness, 0.10119 higher fairness, and 8.89 later FND. Its 1/100 joint-QoS result shows that fairness and staleness alone do not satisfy the delivery constraint.

## Learned residual versus analytic teacher

The learned residual remained extremely close to the analytic mechanism on the untouched cohort:

- delivery difference: -0.0000393, p=0.841;
- stale difference: -0.0000070, p=0.841;
- fairness difference: -0.0000584, p=0.421;
- FND difference: effectively 0, p=0.705;
- packets/J difference: -0.02849, unadjusted p=0.0126.

The packets/J confidence interval excludes zero, so the residual should not be called numerically identical to the teacher. Its small absolute efficiency cost is measurable. The earlier ablation established 95.36% teacher-correction reduction and only 0.194% parameter overhead; together, the evidence supports **operational approximation within a hybrid controller**, not replacement of all analytic logic.

## What succeeded

1. The final cohort was frozen before execution and remained disjoint from every earlier cohort.
2. All 500 matched rollouts completed, all policy/rank pairs were present, and all FND events were observed.
3. The primary controller cleared its preregistered 90/100 joint-QoS integrity floor with 99/100 passes.
4. Multiplicity correction was applied to the full prespecified family rather than only favorable comparisons.
5. The final result agrees with the mechanism ablation: suppressing aggressive upper-band service preserves fairness/staleness reliability but costs some delivery and lifetime.

## Challenges and their resolution

- **Windows multiprocessing permission:** the first command failed before worker creation with `WinError 5`. No cohort results were generated. The exact same hashed command was rerun with local multiprocessing permission.
- **Long runtime:** the full evaluation took 846.2 seconds. Five concurrent workers used three Torch threads each, for a 15-thread CPU contract. Process checks confirmed all workers accumulated CPU time with stable memory.
- **Pseudo-replication risk:** five ranks share one seed. The evaluator collapses ranks to a seed mean before inferential tests.
- **Comparator compatibility:** published numerical tables use incompatible simulators. This evaluation uses structural adaptations inside one common simulator and labels them as adaptations.
- **Mixed objective directions:** no composite score was invented after seeing the data. Delivery, staleness, fairness, FND, and energy efficiency remain separate prespecified endpoints.

## Paper-safe claims

Supported:

- HTA-MAC provides reliable joint-QoS satisfaction under the frozen primary scheduled EH-WSN simulator.
- It trades a modest amount of delivery, FND lifetime, and packets/J relative to tuned energy-proportional for lower staleness and substantially higher service fairness.
- A 225-parameter permutation-equivariant residual ranker closely approximates the globally coupled analytic removal order with a 3.7 KB checkpoint.
- Removing upper-band control increases delivery but harms FND, fairness, and efficiency, establishing that the controller implements a real allocation trade-off.

Not supported:

- universal lifetime or energy-efficiency superiority;
- direct numerical superiority over values published under different simulators;
- a fully neural end-to-end constrained policy;
- removal of the frozen HEART-CH schedule or extension to routing optimization.

## Recommended manuscript sequence

1. Frame the contribution as **risk-aware, fairness-preserving joint-QoS MAC control under a frozen CH schedule**.
2. Put the 20-seed matched comparison in the main Results section.
3. Put the learned-versus-teacher compatibility and no-upper-removal mechanism ablation immediately after it.
4. Present the energy-proportional comparison as a Pareto trade-off, not a defeat to hide: +0.050 fairness and -0.00469 staleness at costs of -0.00576 delivery and -7.33 FND rounds.
5. Keep direct cross-paper numbers in a related-work assumptions table, clearly separated from matched-simulator empirical claims.
6. If one more experiment is affordable, estimate a Pareto curve by freezing three predeclared upper-band targets and evaluating them on a new development cohort; do not reopen seeds 3700-3719 for tuning.

## Reproduction

```powershell
python -B experiments/evaluate_step3_final_matched_baselines.py `
  --contract config/step3_final_matched_baseline_evaluation_v1.json `
  --output outputs/phase3/step3_final_matched_baseline_evaluation_v1/summary.json

python -B experiments/render_step3_final_evidence.py `
  --summary outputs/phase3/step3_final_matched_baseline_evaluation_v1/summary.json `
  --output-dir outputs/phase3/step3_final_matched_baseline_evaluation_v1/publication
```

Publication artifacts:

- `publication/table_final_aggregate_metrics.csv`
- `publication/table_final_paired_inference.csv`
- `publication/figure_final_policy_metrics.png`
- `publication/figure_final_paired_differences.png`
- `publication/FINAL_RESULTS_CAPSULE.md`

## Final assessment

The project now has a coherent, defensible empirical story: the model is not the longest-lived allocator, but it is dramatically more reliable than the literature-adapted sleep/layer and fixed-frame baselines at satisfying the complete QoS contract, and it improves fairness and staleness relative to a strong tuned energy heuristic. The negative lifetime result should be stated plainly. Doing so strengthens the paper because the contribution becomes an evidenced Pareto trade-off with reproducible statistics rather than an implausible across-the-board win.
