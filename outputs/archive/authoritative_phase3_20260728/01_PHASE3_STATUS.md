# HTA-MAC Phase 3 status

Status: **comparison layer implemented; Phase 4 blocked** on 2026-07-28.

The structural Phase 3 gate passed: all policies share one interface and one
scheduled environment, 35/35 paired pilot runs completed, no metric was NaN,
and the idle-off static compatibility median was within the prespecified 20%
diagnostic band. The current result is not publication-ready because the pilot
found both a stronger adapted competitor and extensive right censoring.

## Implemented policies

1. static equal TDMA;
2. energy-proportional allocation;
3. harvest-proportional allocation;
4. S2A2MAC-adapted alternating-cluster sleep with per-node 1/2/3 active layers;
5. FFSS-adapted one-slot feasible-first fixed-frame allocation;
6. frozen HTA-MAC Branching C51 checkpoint;
7. seeded random-budgeted diagnostic, explicitly not a literature baseline.

All implement `MACPolicyInterface`, run with `T=24` and `n_max=3`, suppress
allocations when the scheduled CH is dead, and use the identical seed-specific
frozen HEART-CH schedule and embedding sequence.

## Correctness repair discovered during inspection

The first pilot exposed that the new MAC environment harvested energy for dead
nodes, allowing them to revive. Upstream HEART-CH samples harvest and advances
HMM states only for alive nodes, so death is absorbing. The MAC environment was
corrected accordingly, an absorbing-death regression test was added, and the
full Phase 1 gate was rerun successfully before repeating Phase 3.

The invalid pilot is retained under
`outputs/phase3/paired_pilot_5seed_superseded_revival_bug/`. It must never be
used for results.

Revalidated Phase 1 evidence remained unchanged:

```text
ENERGY_TRACE_MAX_ERROR_J=0.000e+00
DETERMINISTIC=True
BUDGET_VIOLATIONS=0/1000
T_FND_IDLE_ON_OFF_MEDIAN=127.0/920.0
AUTHORITATIVE_PHASE_1_GATE=PASS
```

## Five-seed paired pilot

Command:

```powershell
python -B experiments\run_phase3_pilot.py `
  --seeds 3100,3101,3102,3103,3104 `
  --horizon 3000 --run-name paired_pilot_5seed
```

Gate output:

```text
PRIMARY_RUNS=35/35
FAILURES=0
ALL_METRICS_FINITE=True
STATIC_IDLE_OFF_MEDIAN_FND=927.0
RELATIVE_GAP=0.1577321461
PHASE3_STRUCTURAL_GATE_PASS=True
```

Selected median +/- IQR development results follow. They are not inferential
claims because only five seeds were run.

| Policy | T_FND | T_HND | Throughput | Idle J | Fairness | Packets/J |
|---|---:|---:|---:|---:|---:|---:|
| Static equal | 132 +/- 4 | 163 +/- 3 | 13,247 +/- 256 | 45.137 +/- 0.511 | 0.9221 +/- 0.0276 | 240.53 +/- 3.07 |
| HTA-MAC | 141 +/- 16 | censored in 4/5 | 18,385 +/- 54 | 44.307 +/- 0.327 | 0.9129 +/- 0.0051 | 311.98 +/- 2.29 |
| S2A2MAC-adapted | 201 +/- 5 | 354 +/- 23 | 22,637 +/- 481 | 46.032 +/- 0.222 | 0.8403 +/- 0.0087 | 381.51 +/- 5.80 |
| Random diagnostic | 120 +/- 7 | 193 +/- 11 | 15,377 +/- 140 | 44.135 +/- 0.281 | 0.9351 +/- 0.0047 | 283.46 +/- 1.68 |

Paired diagnostic deltas:

- HTA minus static: median T_FND `+13`, throughput `+5,187`; HTA won
  throughput on all five seeds and T_FND on four of five.
- HTA minus random: median T_FND `+24`, throughput `+2,991`; HTA won both on
  all five seeds, but lost fairness on all five.
- HTA minus S2A2MAC-adapted: median T_FND `-55`, throughput `-4,058`; HTA lost
  both on all five seeds, but won fairness on all five.

## Why Phase 4 is blocked

1. The current HTA checkpoint does not beat the S2A2MAC adaptation on T_FND,
   throughput, or energy efficiency in any pilot seed. Reward/policy diagnosis
   must return to Phase 2 before spending 210 final runs.
2. Four of five HTA HND values are right-censored. A raw paired Wilcoxon test on
   these HND values would be invalid; a censor-aware endpoint or schedule policy
   is required first.
3. Every run is censored when the frozen HEART-CH schedule selects no CH at
   1,633-1,730 rounds, despite the requested 3,000-round horizon. The last frame
   is correctly never repeated.
4. Primary-source verification disproved the planned S2A2MAC novelty sentence.
   S2A2MAC already differentiates per-node active periods using residual energy
   and load. HTA-MAC must instead differentiate itself through hybrid
   harvest-trajectory conditioning, learned allocation, and the explicit idle
   accounting model.
5. FFSS ordering is not representable in the round-level action model; only the
   documented FFSS-adapted policy can be claimed.

## Evidence

- Raw trials: `outputs/phase3/paired_pilot_5seed/raw_trials.csv`
- Pilot summary: `outputs/phase3/paired_pilot_5seed/summary.json`
- Paired audit: `outputs/logs/phase3_pilot_audit.json`
- Static idle-off compatibility: `outputs/phase3/paired_pilot_5seed/static_idle_off_compatibility.csv`
- Baseline provenance: `BASELINE_PROVENANCE.md`
- Policy configuration: `config/phase3.yaml`

Final regression run: `19 passed` using
`python -B -m pytest validation -q -p no:cacheprovider`.
