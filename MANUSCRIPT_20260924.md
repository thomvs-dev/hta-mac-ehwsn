# IEEE IoT Journal manuscript and statistical supplement — 24 September 2026

The [ten-page manuscript](output/pdf/iotj_fragment_recovery_20260924_statistics.pdf)
contains the component model, equations, six tables, simulation figures and
15 references. [Editable source package](output/pdf/iotj_fragment_recovery_20260924_statistics_source.zip)
and [source/analysis files](paper/iotj_fragment_20260924/) are provided.
Author details remain pending; this is a draft, not a journal acceptance claim.

The [statistical supplement](paper/iotj_fragment_20260924/STATISTICAL_REVIEW.md)
uses independent paired seeds (64,64,32), not all 64,512 episodes as independent
samples. It reports 22 two-sided paired tests with global Holm correction,
pointwise confidence intervals, paired effect sizes and exact sign-test checks.
All 32 shared-head seed comparisons favor block ACK on both endpoints against
both matched baselines. These post hoc mean seed-ratio tests supplement the
original aggregate bootstrap gates; they do not replace them or establish
superiority over external papers. Failed prefix and guard outcomes remain failed.

Reproduce and verify the supplement:

```powershell
python -B paper/iotj_fragment_20260924/analyze_statistics.py
python -B paper/iotj_fragment_20260924/verify_statistics.py
```

The [next study plan](reports/FRAGMENT_NEXT_STUDY_PLAN_20260924.md) proposes
completion-aware admission and adaptive feedback, beginning with bottleneck
diagnostics, then frozen ablations and conditional matched-protocol validation.
It has not been implemented or run. Original V1 and Gate A/B remain unchanged.
