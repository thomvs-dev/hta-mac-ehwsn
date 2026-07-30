# Phase 0 inspection status

Status: **FAIL — do not begin Phase 1**

Authoritative run:
`outputs/logs/phase0_gate_20260727T210959Z.json`

- Frozen upstream commit: `d96abce25237feb2b6d6c660f6b4d605feb94330`
- Trials: 30 independent seeded episodes (`1000` through `1029`)
- Crashes/NaNs: 0
- Fresh T_FND: `1100.6 ± 44.18` rounds (population standard deviation)
- Fresh T_FND median ± IQR: `1100.0 ± 58.25` rounds
- Locked reference: `1191.3 ± 40.0` rounds
- Absolute mean difference: `90.7` rounds
- Predeclared reproduction threshold: `23.61` rounds
- Reproduction decision: fail

The Stage 1 artifact contains the trained eight-state solar HMM parameters but
no trained four-state thermal HMM parameters. The upstream simulator constructs
thermal transition/emission parameters from Python defaults. Consequently, the
HMM-artifact portion of the Phase 0 gate also fails.

No idle-energy, MAC environment, budget projection, or Branching DQN code has
been implemented.


## Additional provenance audit

Re-running the exact manuscript seed set, seeds 42 through 61, produced
`T_FND = 1083.4 ± 37.13`, not the locked value. The checkpoint's embedded
10-episode evaluation metadata reports `mean_t_fnd = 1122.1`. The locked JSON
contains summaries only and has no raw per-trial data, config hash, or source
commit.

The upstream thermal defaults have been frozen as
`core/hmm/thermal_auxiliary_params.npz` for parameter identity. Its provenance
manifest records `trained: false`; this does not resolve the missing trained
thermal-HMM gate requirement.
