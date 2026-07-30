# HTA-MAC

HTA-MAC is a bounded intra-cluster MAC research project built on a frozen
HEART-CH cluster-head policy. Routing and CH retraining are out of scope.

## Current verified boundary

- Phase 0: pass under the corrected reproduced baseline.
- Phase 1: pass after the instructor-directed pre-training corrections.
- Phase 2: fixed-cluster training gate passed; superiority is not established.
- Phase 3: comparison layer implemented; Phase 4 is blocked by pilot evidence.

Authoritative status:

- `PHASE0_STATUS.md`
- `PHASE1_STATUS.md`
- `PHASE2_STATUS.md`
- `PHASE3_STATUS.md`
- `BASELINE_PROVENANCE.md`
- `PRE_PHASE2_DECISION_CLOSURE.md`

Canonical verification:

```powershell
$env:PYTHONPATH='F:\WSN\matlab\stage2\hta-mac'
python -B validation\phase0_gate.py
python -B validation\run_phase1_gate.py
python -B experiments\train_phase2_fixed_cluster.py --episodes 500 `
  --max-steps 150 --run-name authoritative_500ep_seed2100
python -B experiments\run_phase3_pilot.py `
  --seeds 3100,3101,3102,3103,3104 --horizon 3000 `
  --run-name paired_pilot_5seed
python -B -m pytest validation -q -p no:cacheprovider
```

Expected terminal results:

```text
PHASE_0_CORRECTED_GATE=PASS
AUTHORITATIVE_PHASE_1_GATE=PASS
PHASE2_GATE_PASS=True
PHASE3_STRUCTURAL_GATE_PASS=True
19 passed
```

The immutable pre-Phase-2 evidence is archived under:

```text
outputs/archive/authoritative_pre_phase2_20260728/
```

## Non-negotiable scope

- shared exogenous schedules from the frozen HEART-CH checkpoint;
- no CH retraining;
- no routing changes;
- state-conditioned transition probabilities, not posterior claims;
- manuscript rectified-Gaussian forecast moments;
- fixed synthetic thermal auxiliary marked `trained=false`;
- paired comparisons against the reproduced baseline, not the unreproduced
  manuscript number.
