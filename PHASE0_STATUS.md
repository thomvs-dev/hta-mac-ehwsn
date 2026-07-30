# HTA-MAC Phase 0 current status

Status: **PASS under the corrected empirical foundation**.

The original strict reproduction gate failed because the released checkpoint
did not reproduce the manuscript value `1191.3 +/- 40.0`. That failure is
preserved in `PHASE0_ORIGINAL_GATE_STATUS.md`.

The user-authorized executable baseline is:

```text
30 trials, seeds 1000-1029
T_FND mean = 1100.6
population std = 44.18189674516023
median = 1100.0
IQR = 58.25
```

The thermal model is a fixed synthetic auxiliary derived from upstream
defaults and is explicitly marked `trained: false`.

Authoritative contract and evidence:

- `config/phase0_acceptance.yaml`
- `outputs/logs/phase0_corrected_gate.json`
- `validation/phase0_gate_corrected.py`

All HTA-MAC comparisons must use the reproduced baseline. The manuscript value
may be reported only as the original paper's result, not as a reproduced
measurement.
