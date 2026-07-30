# HTA-MAC Instructor Action and Research Integrity Report

**Report date:** 2026-07-31 (Asia/Calcutta)  
**Repository:** `F:\WSN\matlab\stage2\hta-mac`  
**Frozen pre-training commit:** `f45f02b` (`Freeze schema-v2 branching sweep and Phase 4 analysis plan`)  
**Tag:** `phase2-schema-v2-pretraining`  
**Scope:** Frozen HEART-CH CH selection plus intra-cluster MAC only. Routing, Pointer Networks, and CH retraining remain excluded.

## 1. Executive status

The instructor feedback was not applied as cosmetic manuscript editing. It exposed two evidence-invalidating implementation issues and several missing experimental controls. Those were corrected before launching Phase 4:

1. The old frozen CH schedule generator was not reproducible from its recorded seed because NoisyNet inference was not preceded by a PyTorch seed reset. It also mislabeled inherited upstream termination as “no CH selected.” Schedule schema v2 now seeds the complete stack and reports the true inherited termination cause.
2. The old `BranchingDuelingC51` was a weight-tied local scorer, not the claimed Tavakoli-style shared global decision module with separate node action branches. It is now retained only as `legacy_weight_tied`. The corrected primary architecture and an actual independent-DQN ablation have been implemented and tested.

Consequently, the earlier budget-8 Phase 2 checkpoint and its schema-v1 Phase 3 pilot are explicitly **historical development evidence**, not authoritative final evidence. No final superiority result is claimed.

Current verified state:

- Validation: **38 passed**, 0 failed; 93 upstream-library deprecation warnings.
- Locked-artifact verifier: **28 checks**, **0 failures**.
- Frozen schedule repeatability: seed 3100 produced identical signature `10a571b3e53165075e61b3ee43a4a3b32d24269fa0900095c0e478283ee16312` on two fresh schema-v2 generations.
- Registered training plan: **18 runs** (15 primary + 3 architecture ablation), dry-run manifest generated successfully.
- Phase 4: **blocked** until the corrected Phase 2 gates pass and all registered checkpoints are archived.

## 2. What was implemented

### 2.1 Dedicated repository and immutable pre-training snapshot

A dedicated Git repository was initialized in `hta-mac`. The earlier validated pilot state was committed as `af3289c` and tagged `phase3-heldout-pilot-complete`. After applying the instructor corrections, the pre-training state was committed as `f45f02b` and tagged `phase2-schema-v2-pretraining`.

Reason: future checkpoints must identify the exact local code revision, not an unrelated parent repository. Both Phase 2 and Phase 3 Git-hash helpers were corrected to resolve the `hta-mac` repository itself.

### 2.2 Schedule ceiling and determinism audit

Implemented in:

- `core/ch_selection/frozen_schedule_full.py`
- `validation/test_frozen_schedule_provenance.py`
- `SCHEDULE_CEILING_AND_DETERMINISM_AUDIT.md`
- `outputs/logs/schedule_ceiling_audit_v2.json`

Findings:

- A fresh seed-3100 schedule covers 1640 rounds and ends with 9 alive nodes.
- The upstream episode reports T_FND 1178 and T_HND 1539.
- Termination cause is `alive_fraction_below_death_threshold`, inherited from HEART-CH's 10% alive threshold.
- The approximately 1700-round ceiling is therefore not an arbitrary cache/generator cap. Continuing past it would change the frozen upstream environment contract.
- No stale final schedule frame is replayed. Downstream observations are right-censored at genuine schedule termination.

Decision: preserve the inherited termination and use censor-aware analysis. Do not fabricate a 3000-round CH schedule by repeating or synthesizing frames.

### 2.3 Schedule schema v2

Schema v2 now:

- seeds Python, NumPy, and PyTorch before frozen NoisyNet CH inference;
- includes the schema version in schedule cache names;
- records coverage, terminal alive count, upstream episode metrics, stop reason, and termination cause;
- passes a fresh repeated-generation signature test.

Decision: all schema-v1 HTA checkpoint/pilot evidence is superseded for final claims. Pairing within the old pilot was internally consistent, but the schedule was not regenerable from the recorded seed alone.

### 2.4 Corrected Branching Dueling architecture

Implemented in:

- `agents/architectures.py`
- `agents/branching_dqn.py`
- `baselines/policies.py`
- `experiments/train_phase2_dynamic_curriculum.py`

Primary `shared_branching` model:

- zero-pads the 100-node global state and appends the active-node mask;
- processes it through one shared global decision module;
- uses 100 node-indexed dueling distributional action heads;
- produces one discrete slot-count distribution per global node;
- applies the same queue caps and greedy cluster-budget projection as before.

This network has **2,842,811 online parameters** under the frozen configuration.

Independent-DQN ablation:

- uses 100 separately parameterized local dueling C51 networks;
- has no shared global decision module and therefore no cross-node feature context before projection;
- has **5,622,700 online parameters**.

Tests verify the intended distinction: changing node 2's input changes node 1's Q-values in the shared model, but leaves node 1 unchanged in the independent model.

Legacy behavior:

- old checkpoints lacking an architecture field load only as `legacy_weight_tied`;
- a legacy checkpoint cannot initialize a corrected shared/independent training run;
- new Phase 3 evaluation supplies full global node identity to shared/independent checkpoints rather than reindexing each cluster's members into the first branch heads.

Reason: training the original implementation would not support the paper's architectural claim or the requested architecture ablation.

The trajectory-Q inspection gate was also corrected after architectural review. It now changes S1 to S8 on the **same global node and same action head**, at fixed normalized residual energy, while holding the rest of the global state constant. The old two-row check could confound state sensitivity with different head parameters.

### 2.5 HMM taxonomy correction

Reward categorization now follows the upstream HEART-CH taxonomy exactly:

- high/good: one-based S6 and S8, stored as zero-based `{5, 7}`;
- avoid/critical: one-based S1, S4, and S7, stored as `{0, 3, 6}`.

The synthetic thermal auxiliary remains in the observation and hybrid forecast moments, but it receives no invented categorical reward label because no validated thermal state taxonomy exists.

Tests exhaustively verify the mapping over all eight solar states.

### 2.6 Delivery, drops, and residual-energy fairness

The environment/evaluator now records:

- packets generated, including the initialized queue population;
- delivery ratio;
- stale-drop ratio;
- death-time packet drops after subtracting service in the death round;
- overflow drops;
- residual-energy Jain fairness including dead-node zeros;
- residual-energy coefficient of variation;
- mean and minimum residual energy.

Reason: throughput alone hid the severe QoS loss in the old low-budget pilot. The lifetime benefit must be presented as a Pareto tradeoff against delivery/freshness, not as unconditional improvement.

### 2.7 Baseline tuning on protected development seeds

Added `experiments/tune_phase3_baselines.py`. Tuning used only schema-v2 seeds 2500–2504, separate from both training seeds and future Phase 4 seeds. Seventy runs completed (14 configurations x 5 seeds).

Frozen selections in `config/phase3.yaml`:

- energy-proportional exponent: 2.0;
- harvest-proportional exponent: 2.0;
- S2A2MAC adaptation: energy weight 0.25, load weight 0.75;
- FFSS adaptation: margin weight 1.0, queue weight 0.0.

Raw tuning CSV SHA-256: `0834fbce9955bd3a0e0f0b2fff6735b60555d4b551a049c1a6d7243c064fffd4`.

Decision: baseline parameters are now frozen. Phase 4 seeds cannot retune them.

### 2.8 Random policy promoted to formal floor

The random-budgeted policy remains named `random_budgeted_diagnostic` for artifact compatibility, but its comparison role is now explicitly `formal_stochastic_floor`. Phase 3/4 summaries include this role and paired statistical comparisons.

Reason: a learned method that cannot reliably beat a budget-matched random allocation is not a useful learned scheduler, regardless of reward convergence.

### 2.9 Statistical effect sizes

Added reusable functions in `core/paired_statistics.py` for:

- two-sided paired Wilcoxon signed-rank testing;
- median paired difference;
- paired Hodges-Lehmann shift using Walsh averages;
- matched-pairs rank-biserial correlation;
- wins, ties, and losses;
- defined all-tie behavior (`p=1`, effect 0, statistic undefined).

Non-lifetime endpoints use this shared implementation. Lifetime endpoints remain censor-aware using common-horizon restricted event-free time and Kaplan-Meier summaries.

### 2.10 Hashed Phase 4 preregistration

Created:

- `PHASE4_PREREGISTRATION.md`
- `PHASE4_PREREGISTRATION.sha256`

Frozen digest: `30e64870ed2d4614051021853915e7ddef5daec509159ecb2c9c11aaca2c1790` (9604 bytes).

The plan freezes:

- budgets `{8,12,16,20,24}`;
- training seeds `{2299,3299,4299}`;
- development schedules `{2300..2304}`;
- Phase 4 test schedules `{4000..4029}`;
- 500 episodes, maximum 300 steps;
- three independently initialized models per budget;
- a budget-12 three-seed independent-DQN ablation;
- FND-free restricted time and delivery ratio as co-primary endpoints;
- Holm correction for the 20 confirmatory budget/endpoint/static-or-S2 tests;
- effect sizes, failure handling, Pareto rules, censoring, and artifact integrity.

For HTA-MAC, test-seed metrics are averaged across the three training replicates before paired inference. This preserves the environment seed as the inferential unit instead of treating three networks on one trajectory as independent samples.

### 2.11 Registered sweep executor

Added `experiments/run_phase2_registered_sweep.py`. Its dry run prints:

- `REGISTERED_RUNS=18`
- `GATE_PASS_RUNS=0`
- `REGISTERED_SWEEP_COMPLETE=False`

It refuses metadata-mismatched output directories, does not substitute failed seeds, and stops after a gate failure unless explicitly asked to continue for failure characterization.

### 2.12 Manifest verification

`validation/verify_manifest.py` verifies upstream commit, frozen checkpoint, solar HMM, synthetic thermal auxiliary, historical HTA artifacts, report/preregistration/novelty hashes, and archived file sizes/hashes.

Latest output:

- `MANIFEST_CHECKS=28`
- `MANIFEST_FAILURES=0`
- `ARTIFACT_MANIFEST_PASS=True`

## 3. Novelty audit and manuscript corrections

Created `NOVELTY_AND_CLOSEST_WORK_AUDIT.md` after inspecting primary-source pages.

Material corrections:

1. The supplied sentence “S2A2MAC applies one HMM-derived active-period rule per cluster” must be removed. Movva et al.'s primary abstract says the HMM controls a **node's** active period adaptively.
2. “First hybrid-harvest per-node TDMA slot allocation” is not defensible. Gong et al.'s 2020 SHR-TDMA uses hybrid sources, a Markov-derived arrival characterization, and per-node optimal slot assignment.
3. “No per-node intra-cluster RL slot allocation” is not defensible. Eriş et al. (Sensors 2024) use cooperative independent Q-learning for per-node intra-cluster slot choices in an EH-UASN.
4. The universal idle-listening absence claim is narrowed to a verified implementation statement: HEART-CH omits an explicit idle term; HTA-MAC adds it consistently for all compared policies.
5. The underwater RL work must not be dismissed as merely current-energy reactive without stronger full-text evidence.

Current defensible differentiation:

> Among the audited clustered EH-WSN MAC methods, HTA-MAC is distinguished by a centralized Branching Dueling Q formulation that selects a discrete per-node slot count under a cluster budget from node-specific solar-and-thermal HMM trajectory features, while keeping the upstream CH schedule fixed for paired causal attribution.

This remains test-contingent and is not phrased as a worldwide first claim.

## 4. Decisions and reasons

| Decision | Reason |
|---|---|
| Keep full-data-slot idle accounting as primary and 100-bit header accounting as sensitivity | Both are dimensionally defensible; reporting both exposes magnitude sensitivity rather than hiding it. |
| Preserve inherited HEART-CH schedule termination | Extending it would modify the frozen upstream system and confound attribution. |
| Use shared schedule replay | Keeps CH selection exogenous and isolates the MAC effect. |
| Supersede schema-v1 checkpoint/pilot | Seed provenance was incomplete and the architecture did not match the claimed method. |
| Train all five budgets independently from scratch | Warm starts would couple arms and preserve the superseded reward taxonomy/architecture. |
| Aggregate three model replicates within each environment seed | Captures training stochasticity without pseudoreplication. |
| Report a Pareto frontier | Lifetime and delivery move in opposing directions; a post hoc scalar score would hide the tradeoff. |
| Keep random as a formal floor | Establishes whether learning adds value beyond budget-matched stochastic allocation. |
| Add independent DQNs at budget 12 | Directly tests whether the shared decision module contributes beyond local learners under a fixed resource regime. |
| Avoid absolute “first/no paper” claims | Targeted source inspection found direct counterexamples to the broader language. |

## 5. Failed or superseded work retained transparently

- Historical reproduced HEART-CH baseline: `1100.6 ± 44.18` from the earlier 30-trial reproduction, not the manuscript's unverified `1191.3 ± 40.0` provenance.
- Historical schema-v1 budget-8 checkpoint SHA-256: `0ef29efaff04ec1cb652c84a432a53bd0c41d7c68dc9decfcadf9c277247c2ff`.
- Historical schema-v1 held-out raw CSV SHA-256: `7529e8ccb99f35f6e0bcc6552b62bd9c5b8de95b2e584548e17c51fd6d076438`.
- A corrected-taxonomy but legacy-architecture 100-episode probe was interrupted after the architecture mismatch was discovered. It is ignored as a development probe and is not evidence.
- Five-episode shared-branching and one-episode independent-DQN execution smokes completed without tensor/checkpoint errors. They are smoke tests only and fail convergence/stability by construction.

## 6. Current run and what remains

The 100-episode from-scratch shared-branching budget-12 diagnostic completed in 1554.6 s on CPU. It printed `ALWAYS_SLEEP_COLLAPSE=False`, `REWARD_PATHOLOGICAL_DOMINATION=False`, `FULL_CURRICULUM_SEEN=True`, and greedy mean target packets `1261.0`. It correctly did not pass the Phase 2 gate: only two deterministic snapshots existed and the reward-convergence test failed at episode 100. The two snapshot metrics were FND-free `141.00 -> 141.96`, throughput `10375.00 -> 10342.88`, and fairness `0.436766 -> 0.446520`. A post-run corrected same-node/same-head check produced S8-S1 maximum Q difference `0.00214028` and `DIFFERENTIATED=True`. This remains a development diagnostic, not an authoritative checkpoint.

Remaining sequence:

1. **Completed:** 100-episode collapse diagnostic; no sleep collapse or non-finite loss, but no convergence/gate pass at this short horizon.
2. If it does not collapse, execute the registered 18-run Phase 2 sweep from commit `f45f02b`.
3. Archive every run, including gate failures; update the artifact manifest and hashes.
4. Do not start Phase 4 if any primary budget arm intended for reporting lacks its three registered gate-passing models. Report instability rather than replacing seeds.
5. Run the 30 paired Phase 4 seeds only after manifest verification and checkpoint admission.
6. Apply the preregistered censor-aware statistics, Holm adjustment, effect sizes, random-floor comparison, and Pareto analysis.
7. Complete idle-bit-time, solar-only/hybrid, reward-weight, `n_max`, and lifetime-bound ablations in Phase 5.
8. Update manuscript contributions to match only observed results and the narrowed novelty audit.

## 7. Open doubts and risks

1. **CPU cost:** CUDA is unavailable. The corrected shared model has 2.84M parameters and the independent ablation 5.62M. The completed 100-episode diagnostic required 1554.6 s including two deterministic evaluations and the final evaluation. A 500-episode shared run is therefore expected to take roughly 1.5-2.5 hours, and the complete 18-run sweep may require roughly 30-45 CPU-hours if no run fails early. This is a scheduling constraint, not permission to reduce registered episodes after observing results.
2. **Training stability:** the corrected architecture is newly implemented. Passing tensor-level smokes does not prove 500-episode stability; the 100-episode diagnostic is the first meaningful collapse check.
3. **Independent-DQN feasibility:** the ablation is larger and may be slower or less stable. Failure is itself reportable architecture evidence and cannot be replaced by a lighter unregistered model.
4. **Thermal realism:** thermal parameters remain synthetic upstream defaults. Hybrid-source experiments test mechanism separation, not real-world thermal forecasting.
5. **Novelty completeness:** the audit is targeted rather than PRISMA/systematic. A final database-indexed search and full-text mechanism audit is still required before submission.
6. **FFSS/S2A2MAC fidelity:** both remain adaptations where the source mechanism cannot be represented exactly in the round-level environment. The limitations must remain visible in the paper.
7. **Schedule censoring:** inherited upstream termination may prevent estimation of median HND for some policies. Restricted event-free time and event counts remain the primary defensible treatment.

## 8. Reproduction commands

```powershell
cd F:\WSN\matlab\stage2\hta-mac
python -B -m pytest validation -q -p no:cacheprovider
python -B validation\verify_manifest.py
python -B experiments\run_phase2_registered_sweep.py --dry-run
python -B experiments\run_phase2_registered_sweep.py
```

The last command is intentionally long-running and fail-fast. It must be run from the tagged pre-training repository state or a clearly recorded descendant commit containing only evidence/runner corrections.