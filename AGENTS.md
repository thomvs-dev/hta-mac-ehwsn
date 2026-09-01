# HTA-MAC repository instructions

These instructions apply to the complete repository.

## Start here

Before changing code or running experiments, read:

1. `HTA_MAC_NEW_CHAT_EXECUTION_HANDOFF_20260901.md`
2. `releases/HTA_MAC_PAPER_BASELINE_V1_20260901/README.md`
3. `releases/HTA_MAC_PAPER_BASELINE_V1_20260901/PAPER_CLAIMS.md`
4. `HTA_MAC_DELIVERY_EFFICIENCY_IMPROVEMENT_PLAN_20260901.md`

The frozen manuscript baseline is tag `paper-baseline-v1-20260901`, commit
`11ef88c336a18833b8511b30dba857ae8b831086`. Never overwrite its checkpoint,
configs, evidence, manuscript, or release directory.

## Scientific scope

- HTA-MAC learns bounded intra-cluster MAC allocation only.
- HEART-CH cluster-head schedules are exogenous and frozen.
- Do not add cluster-head retraining, routing optimization, or Pointer Networks
  to the claimed method without explicit user authorization and a new study.
- Preserve queue caps, the global action budget, dead-node masking, absorbing
  death, and separated member-TX versus CH receive/aggregate/forward accounting.
- Raw physical metrics must remain unscaled even when learning rewards are
  transformed.

## Evidence discipline

- Never fabricate, interpolate, or silently repair results.
- A failed preregistered gate must be repaired or reported; never weaken it
  after seeing the data.
- Keep development, pilot, and confirmation cohorts distinct.
- Seeds 3900--3919 are opened confirmation seeds and are prohibited for all
  future tuning, checkpoint selection, and early stopping.
- The cap-corrected energy audit supersedes the earlier energy-proportional
  ranking. Do not restore the uncapped comparison as the paper headline.
- Cross-paper percentages are context only unless simulator, topology, traffic,
  energy model, horizon, policies, and endpoints are genuinely matched.
- Call the online primal-dual comparator author-constructed. It is not a
  reproduction of PPO-Lagrangian or CPO.
- The PVGIS irradiance trace is real; the network remains simulated.

## Current result boundary

At the corrected 100-node reference point, HTA-MAC V1 records delivery 0.42770,
stale loss 0.02476, Jain fairness 0.94511, RMST 128.28, and 225.77 packets/J.
The corrected residual-energy heuristic records delivery 0.44591, stale loss
0.04838, fairness 0.87166, RMST 149.32, and 242.36 packets/J.

The defensible V1 contribution is a fairness/staleness trade-off, not universal
superiority. The checkpoint-producing summary also records failed development
gates; preserve that fact.

## V2 execution rules

- Work on a new `codex/v2-*` branch and use new V2 config, checkpoint, output,
  and report paths.
- Do not train immediately. Execute Phase A of the improvement plan first: a
  CPU-only marginal service/energy mechanism audit.
- Gate A must attribute at least 80% of the delivery/efficiency gap to measured
  selection, under-service, or marginal-energy mechanisms before proceeding.
- Gate B requires the deterministic marginal-utility teacher to achieve
  delivery at least 0.450 and packets/J at least 244.8 while satisfying the
  frozen secondary constraints. If it fails, stop neural training and report
  the structural frontier.
- Before training, require permutation, padding, finite-value, exact-budget,
  queue-cap, and unseen-state teacher-agreement tests.
- Add a matched recognized constrained-RL baseline before making a constrained-
  RL advance claim; PPO-Lagrangian or guarded PPO is preferred.
- Use high CPU utilization only across independent seeds/environments. Use a GPU
  for neural minibatch training after the CPU gates pass.

## Verification and repository care

- Run `python -B -m pytest validation -q -p no:cacheprovider` after relevant
  changes. The V1 freeze passed 174 tests on 1 September 2026.
- The working tree may contain user-owned untracked results and a modified
  `paper/refs.bib`. Inspect status and preserve unrelated files.
- Do not commit historical training bundles, logs, scratch directories, or
  downloaded evidence unless explicitly selected for a release.
- Keep `.firecrawl/` ignored.
- Record exact paths, hashes, seeds, horizons, budgets, confidence intervals,
  effect sizes, and gate outcomes in every experiment handoff.

