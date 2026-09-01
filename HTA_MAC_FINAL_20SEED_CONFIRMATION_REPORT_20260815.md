# HTA-MAC final 20-seed confirmation report

**Confirmation date:** 15 August 2026  
**Status:** final confirmation complete  
**Model decision:** retain the frozen source checkpoint; no retraining or
post-confirmation selection is authorized.

## Executive result

The preregistered confirmation succeeded for the claims that were actually
tested:

1. HTA-MAC had higher delivery than energy-proportional in every one of ten
   scenarios. All ten comparisons remained significant after Holm correction.
2. HTA-MAC had substantially longer lifetime than energy-proportional in the
   20-node transfer scenario.
3. In the reference scenario, HTA-MAC was non-inferior to the online
   primal-dual baseline under all three frozen practical margins: delivery,
   restricted mean survival time (RMST), and packets/J.
4. Universal lifetime superiority is not supported and is not claimed.

The publishable result is therefore **robust delivery improvement with a
scenario-dependent QoS--lifetime Pareto tradeoff**, not dominance on every
metric.

## Frozen protocol

- Confirmation seeds: `3900--3919` (20 independent paired seed units).
- Inference unit: the seed mean across target-cluster ranks. Cluster ranks were
  not counted as independent statistical samples.
- Horizon: 3,000 rounds.
- Policies: frozen raw HTA-MAC C51, energy-proportional (exponent 4), and the
  frozen online primal-dual allocator.
- Scenarios: reference, 20 nodes, 50 nodes, low/high traffic, low/high harvest,
  half battery, large field, and a frozen PVGIS SRRL 2020 irradiance replay.
- Statistics: paired 20,000-resample bootstrap 95% CIs, paired Wilcoxon
  signed-rank tests, and Holm correction across the ten delivery-robustness
  tests.
- FND handling: event counts plus RMST capped at 3,000 rounds. Every evaluation
  unit observed FND before the cap, so the reported RMST values equal the mean
  observed event-free duration for this confirmation.
- No training, retuning, checkpoint selection, or threshold revision occurred
  after the confirmation seeds were opened.

## Confirmed delivery robustness

The table reports seed-level means. The difference and CI are absolute delivery
ratios for HTA-MAC minus energy-proportional.

| Scenario | HTA-MAC | Energy-proportional | Difference | Bootstrap 95% CI | Holm-adjusted one-sided p |
|---|---:|---:|---:|---:|---:|
| Reference 100 | 0.42770 | 0.36890 | +0.05880 | [0.05639, 0.06112] | 9.54e-6 |
| Nodes 20 | 0.90098 | 0.40009 | +0.50090 | [0.49418, 0.50803] | 9.54e-6 |
| Nodes 50 | 0.37523 | 0.31492 | +0.06031 | [0.05701, 0.06355] | 9.54e-6 |
| Traffic low | 0.87002 | 0.57873 | +0.29129 | [0.28761, 0.29502] | 9.54e-6 |
| Traffic high | 0.29338 | 0.28808 | +0.00530 | [0.00439, 0.00610] | 9.54e-6 |
| Harvest low | 0.42854 | 0.36935 | +0.05919 | [0.05669, 0.06157] | 9.54e-6 |
| Harvest high | 0.42761 | 0.36873 | +0.05887 | [0.05675, 0.06089] | 9.54e-6 |
| Battery half | 0.43513 | 0.37085 | +0.06428 | [0.06052, 0.06809] | 9.54e-6 |
| Field large | 0.43000 | 0.36981 | +0.06019 | [0.05783, 0.06251] | 9.54e-6 |
| PVGIS trace | 0.42831 | 0.36980 | +0.05851 | [0.05614, 0.06074] | 9.54e-6 |

The relative delivery gain was 15.9% in the reference scenario, 125.2% for 20
nodes, 50.3% under low traffic, and 1.84% under high traffic. The high-traffic
effect is small but consistent across all 20 paired seeds.

The identical raw one-sided p-value (`9.5367e-7`) across these comparisons is
not a software artifact: each difference had the same favorable sign in all 20
pairs, which is the minimum exact one-sided sign-pattern probability at n=20.
Holm adjustment across ten tests gives `9.5367e-6`.

## Reference QoS--lifetime tradeoff versus energy-proportional

In the reference scenario:

- Delivery improved by `+0.05880` (95% CI `[0.05639, 0.06112]`).
- Packets/J improved by `+7.114`, a relative gain of 3.25% (95% relative CI
  `[3.13%, 3.38%]`).
- Queue fairness improved by `+0.03982` (95% CI `[0.03392, 0.04640]`).
- The stale ratio was lower by `0.01095` (95% CI `[0.01043, 0.01150]`).
- RMST was **9.71 rounds lower** (HTA-MAC 128.28 versus 137.99; 95% difference
  CI `[-10.25, -9.19]`).

Thus HTA-MAC improves delivery, fairness, stale loss, and energy efficiency, but
energy-proportional delays the first network death in the reference profile.
The manuscript must show this as a Pareto tradeoff.

## Confirmed 20-node lifetime result

For 20 nodes, HTA-MAC improved both service and lifetime relative to
energy-proportional:

- Delivery: 0.90098 versus 0.40009 (`+0.50090`).
- RMST: 150.45 versus 109.25 (`+41.20` rounds).
- RMST bootstrap 95% CI: `[37.35, 44.80]` rounds.
- One-sided paired Wilcoxon p: `4.404e-5`.
- Packets/J: 322.25 versus 183.64.

This supports scenario-specific Pareto superiority at the smaller network size.
It must not be generalized to all node counts: at 50 nodes, HTA-MAC improved
delivery but RMST was 13.08 rounds lower than energy-proportional.

## Comparison with online primal-dual

The online primal-dual controller is a strong constrained, non-neural baseline.
In the reference profile it was numerically better than HTA-MAC:

| Metric | HTA-MAC | Primal-dual | HTA-MAC minus baseline | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Delivery | 0.42770 | 0.43020 | -0.00250 | [-0.00323, -0.00174] |
| RMST | 128.28 | 131.10 | -2.82 | [-3.43, -2.14] |
| Packets/J | 225.77 | 232.45 | -6.68 | [-6.88, -6.50] |

These are statistically detectable differences, so the policies are not
equivalent. However, HTA-MAC passed the three frozen practical non-inferiority
margins:

- Delivery lower CI `-0.00323` > margin `-0.01`.
- RMST lower CI `-3.43` > margin `-6` rounds.
- Relative packets/J lower CI `-2.96%` > margin `-5%`.

The permitted claim is therefore **non-inferior within preregistered practical
margins**, not superiority or equivalence. Also, online primal-dual is not a
PPO/CPO reproduction and must not be described as one.

## QoS-floor interpretation

All three policies passed the calibrated joint-QoS floor in every scenario.
Consequently, the current binary joint-QoS pass metric is not discriminative.
The continuous delivery, stale, fairness, energy-efficiency, and lifetime
effects carry the scientific comparison. The manuscript should report the
100% pass rate for completeness but should not use it as evidence that the
policies perform equally.

## External-trace boundary

The PVGIS scenario uses a real 2020 irradiance trace with a frozen file hash,
but topology, radio energy, queues, and packet delivery remain simulated. It is
valid evidence of robustness to an external harvesting time series. It is not a
field deployment, hardware experiment, or independent third-party replication.

## Claim table for the manuscript

| Claim | Confirmation decision | Permitted wording |
|---|---|---|
| Higher delivery than energy-proportional in reference | Confirmed | HTA-MAC significantly improved delivery under the frozen reference simulator. |
| Delivery robustness across all ten scenarios | Confirmed | Improvement remained positive after Holm correction across the frozen transfer matrix. |
| Longer lifetime at 20 nodes | Confirmed | HTA-MAC improved RMST in the 20-node transfer profile. |
| Universal lifetime superiority | Not supported | Do not claim. Lifetime effects depended on workload and node count. |
| Better than online primal-dual | Not supported | Do not claim. Primal-dual was numerically better in the reference profile. |
| Non-inferior to online primal-dual | Confirmed within frozen margins | State all margins and CIs; do not call this equivalence. |
| Real-world validation | Not supported | The real irradiance trace is replayed inside a simulator. |

## Reproducibility and integrity

- Source checkpoint SHA-256:
  `31dc4bbed0b91ff326066dee24db3d550f6df4a347eaca82c728c4b77103934a`
- Frozen confirmation contract SHA-256:
  `a4341cd63c9bef61319ff70e8cda5b842ca74f521d6499556d93f255474f5548`
- Confirmation runner SHA-256:
  `a1eff260a2db877542081c38ff33af436ce75484df0014887f84d08ff04e62a8`
- Final result SHA-256:
  `e3f6845edcde07d4825ee10c79f33bb875e4f2c76fe40677c6a3a65fc6beb232`
- Final result size: 1,911,338 bytes.
- Multiprocessing smoke: passed on non-confirmation seed 4000.
- Final campaign: 600/600 tasks completed in 479.05 seconds.
- Confirmation seeds are now opened and cannot be reused for model selection or
  revised hypotheses.

## Artifacts

- `config/step4_final_confirmation_v1.json`
- `experiments/run_step4_final_confirmation.py`
- `outputs/phase4/final_confirmation_v1/preconfirmation_smoke.json`
- `outputs/phase4/final_confirmation_v1/final_confirmation.json`
- `HTA_MAC_FINAL_20SEED_CONFIRMATION_REPORT_20260815.md`

## Required next manuscript work

1. Build a delivery--RMST Pareto figure using the seed-level confirmation data.
2. Add the ten-scenario delivery forest plot with paired bootstrap CIs.
3. Report exact margins for primal-dual non-inferiority.
4. Describe the frozen HEART-CH/independent-schedule boundary precisely.
5. Keep the negative workload-retraining ablations as evidence that the retained
   checkpoint was not replaced after confirmation.
6. Do not run further model tuning on seeds 3900--3919.

