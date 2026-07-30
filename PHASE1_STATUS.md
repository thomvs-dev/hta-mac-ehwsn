# HTA-MAC authoritative Phase 1 status

Status: **PASS after instructor-directed corrections** on 2026-07-28.

Definitive command:

```powershell
$env:PYTHONPATH='F:\WSN\matlab\stage2\hta-mac'
python -B validation\run_phase1_gate.py
```

Authoritative printed evidence:

```text
FROZEN_CH_COUNT=5
ENERGY_TRACE_MAX_ERROR_J=0.000e+00
DETERMINISTIC=True
BUDGET_VIOLATIONS=0/1000
HMM_KS_SOLAR_D=0.0116,P=0.511606
HMM_KS_THERMAL_D=0.0157,P=0.169934
T_FND_IDLE_ON_OFF_MEDIAN=127.0/920.0,SHIFT=793.0
IDLE_SENSITIVITY_MEDIANS=127.0/818.0/920.0
IDLE_SENSITIVITY_ORDER_PASS=True
AUTHORITATIVE_PHASE_1_GATE=PASS
```

The sensitivity order is:

1. full 4000-bit data-slot idle listening: median T_FND 127;
2. 100-bit control-header-only idle listening: median T_FND 818;
3. idle disabled: median T_FND 920.

These five-trial values validate the energy mechanism and its sensitivity.
They are not the Phase 4 main comparison.

## Locked structural decisions

- `T=24`, derived as `ceil(1.3 * 18)` from the measured median cluster size.
- Phase 5 T sweep must include the measured contention neighborhood.
- `n_max=3` for the primary configuration.
- One packet is generated per alive node per round.
- Packet TTL is three rounds.
- Static-pilot maximum backlog was four; therefore `q_max=5`.
- The 18-dimensional state uses state-conditioned transition probabilities,
  not Bayesian posteriors.
- Forecast mean/variance use HEART-CH manuscript Eqs. 13-14 for rectified
  Gaussian emissions.
- CH decisions and ST-GCN embeddings use shared exogenous frozen schedule
  replay for paired causal attribution.
- Schedules request 3000 rounds. If the frozen policy can no longer select a
  CH, the MAC trial is explicitly right-censored; the last frame is never
  silently repeated.
- The thermal model remains a fixed synthetic auxiliary with `trained=false`.

## Supporting measurements

- Feature validation: `outputs/logs/harvest_feature_validation.json`
- Cluster contention: `outputs/logs/cluster_contention_analysis.json`
- Queue calibration: `outputs/logs/queue_capacity_calibration.json`
- Gate report: `outputs/logs/phase1_gate.json`
- Immutable archive:
  `outputs/archive/authoritative_pre_phase2_20260728/`

Phase 2 training may now begin, but no learned HTA-MAC results exist yet.
