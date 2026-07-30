# Pre-Phase-2 instructor decision closure

Status: **all eight requested corrections implemented and re-verified**.

## Closure table

| Item | Resolution | Evidence |
|---|---|---|
| File cleanup | Historical gates/configs renamed; canonical Phase 0/1 entry points and configs created; pre- and post-revision evidence archived with hashes | `outputs/archive/` |
| Rectified moments | HEART-CH manuscript Eqs. 13-14 implemented for both sources | `core/hmm/rectified_moments.py` |
| Probability terminology | Active state uses “state-conditioned transition probabilities”; no Bayesian-posterior claim | `outputs/logs/harvest_feature_validation.json` |
| Schedule horizon | Requests 3000 rounds; frozen policy loses valid CHs around rounds 1611-1686 in the five measured seeds; exhaustion now right-censors instead of replaying a stale frame | `core/ch_selection/frozen_schedule_full.py` |
| T selection | Median measured cluster size is 18; primary `T=ceil(1.3*18)=24`; quick T pilot completed | `outputs/logs/cluster_contention_analysis.json`, `outputs/logs/pilot_t_sweep.json` |
| Queue TTL | One packet/alive-node/round, TTL=3, stale drops counted; observed static maximum backlog 4, so `q_max=5` | `outputs/logs/queue_capacity_calibration.json` |
| Idle sensitivity | Primary 4000-bit slot, secondary 100-bit header, and idle-off variants implemented | `outputs/logs/phase1_gate.json` |
| Feature/scope decision | Transition rows reuse exactly; active upstream raw moments differ from paper rectified moments; shared CH schedule and embedding replay are locked for causal attribution | `outputs/logs/harvest_feature_validation.json` |

## New authoritative Phase 1 numbers

Five paired trials:

```text
Full-data-slot idle median T_FND = 127
Header-only idle median T_FND   = 818
Idle disabled median T_FND      = 920
```

The expected order passed:

```text
127 < 818 < 920
```

Other gate checks:

```text
20-round energy-conservation maximum error = 0 J
same-seed determinism = pass
budget violations = 0/1000
solar KS p = 0.511606
thermal KS p = 0.169934
```

These are Phase 1 mechanism checks, not final comparative results.

## Quick T pilot

Three seeds, static equal round-robin, 100-bit header idle model:

| T | Median T_FND | Median delivered packets |
|---:|---:|---:|
| 18 | 995 | 69,745 |
| 22 | 893 | 70,372 |
| 24 | 836 | 68,526 |
| 27 | 813 | 69,425 |
| 30 | 787 | 69,636 |

The monotonic lifetime decrease as T grows is consistent with longer awake
frames and increased idle listening. `T=24` remains selected by the predeclared
cluster-size rule, not by choosing the most favorable pilot outcome. The full
T sweep remains a Phase 5 obligation.

## Important schedule finding

A 3000-round schedule was requested, but the frozen upstream policy eventually
selected no valid CH because its own network had depleted. Measured coverage
was 1611-1686 rounds across seeds 2100-2104.

The implementation now:

- never repeats the last frame;
- reports schedule coverage;
- right-censors a MAC trial that remains active after coverage;
- requires the final evaluation to report the censoring rate.

All authoritative Phase 1 FND measurements occurred before schedule
exhaustion.

## Canonical commands

```powershell
$env:PYTHONPATH='F:\WSN\matlab\stage2\hta-mac'
python -B validation\phase0_gate.py
python -B validation\run_phase1_gate.py
python -B -m pytest validation\test_phase0_foundation.py validation\test_phase1_primitives.py -q -p no:cacheprovider
```

Expected results:

```text
PHASE_0_CORRECTED_GATE=PASS
AUTHORITATIVE_PHASE_1_GATE=PASS
8 passed
```

## Remaining pre-training caution

The environment and gate are now defensible enough to begin Phase 2. The
initial reward weights are still unset and must be calibrated from logged
natural scales. Always-sleep collapse detection remains a hard Phase 2 stop
condition.
