# Locked scope and limitations wording

## Shared upstream schedule

> CH selection is evaluated under a fixed exogenous schedule to enable clean
> causal attribution of MAC-layer effects; joint co-adaptation of CH selection
> and MAC scheduling is left to future work.

## Reproduced baseline

> The original HEART-CH manuscript reports T_FND = 1191.3 +/- 40.0 (Table V).
> Our independent reproduction, using the released checkpoint and an identical
> evaluation protocol over 30 trials, yields T_FND = 1100.6 +/- 44.18. The
> released artifacts do not fully specify the provenance needed to resolve the
> gap. All HTA-MAC comparisons in this paper are therefore reported against our
> reproduced baseline to ensure paired, implementation-consistent comparison.

## Thermal auxiliary

> The thermal-harvesting auxiliary model uses fixed parameters derived from
> the upstream framework's defaults rather than a dataset-trained HMM;
> experiments involving the thermal channel should be interpreted as testing
> the hybrid dual-source mechanism, not validated real-world thermal
> forecasting.

## Probability-feature terminology

Use:

```text
state-conditioned transition probabilities
```

Do not call the active checkpoint environment's transition rows Bayesian
posteriors. A posterior implementation exists in a separate legacy simulator,
but it is not used by the frozen checkpoint environment and is not imported
across that simulator boundary.
