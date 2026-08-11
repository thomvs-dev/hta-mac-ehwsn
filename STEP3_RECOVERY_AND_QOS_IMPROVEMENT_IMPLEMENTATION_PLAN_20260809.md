# HTA-MAC Step 3 recovery and QoS/lifetime improvement plan

**Prepared:** 9 August 2026  
**Status:** implementation plan; no new performance result is claimed  
**Scope:** HTA-MAC slot allocation only; the exogenous CH schedule remains frozen

## 1. Evidence that is frozen now

The downloaded Drive evidence proves that the 100-episode preflight passed and that optimizer seed 6499 completed 500 episodes. Its final checkpoint SHA-256 is `216215e34270c68ca09b16bf4c60c1f86ad1116738f3915148166962aba315ed`. Seed 7499 reached episode 500 in its live log but stopped before final evaluation, summary creation, and Drive synchronization. No usable seed-5499 result is present in the supplied download.

Seed 6499 is a diagnostic baseline, not a selected model. It passed the existing structural curriculum gate, but only 2/50 final training episodes passed the joint target-backlog QoS thresholds. FND was observed in 227/500 training episodes, leaving 273 censored; the observed-only FND median must not be treated as a censor-aware lifetime estimate. No 3,000-round common-policy evaluation was completed.

## 2. Root causes to repair

1. **CH condition is not explicitly observable.** The reward depends on scheduled-CH reserve, forecast harvest, BS distance, and intended forwarding load, but the `phase2d_ttl_cap_v2` policy tensor contains member-local rows and the equivariant network pools active member rows. The scheduled CH is excluded from the active branches. Increasing the reward weight alone cannot reliably teach conditional CH protection.
2. **QoS pressure is too weak and decays.** In the final seed-6499 episode the constraint penalty was about `1.56` against physical reward about `4924`, so the constraint was not load-bearing. The final delivery multiplier was only `0.1867` while the delivery ratio was below its threshold.
3. **The training gate omits QoS feasibility.** `phase2_curriculum_gate_pass` can pass even when final-window target QoS fails.
4. **The reward-balance gate omits `ch_depletion_risk`.** The recovered run shows the term was small, but the gate wiring is still incomplete.
5. **Final weights are saved too late.** The trainer performs the expensive final greedy evaluation before writing the final checkpoint. The notebook synchronizes the run directory only after the subprocess returns. This lost the seed-7499 result after 500 completed episodes.
6. **Training diagnostics and network evaluation are conflated.** Curriculum `fnd_free_steps`, target fairness, and target-backlog QoS are not replacements for the paired 3,000-round Phase 3 network metrics.

## 3. Phase A — provenance and recovery guard

### Implementation

- Create a recovery manifest for every supplied file with size, SHA-256, inferred optimizer seed, artifact role, and validity status.
- Preserve seed 6499 unchanged as `step3_v1_diagnostic_seed6499`; never overwrite or silently promote it.
- Record seed 7499 as `trained_to_episode_500_finalization_incomplete_no_checkpoint` unless its original Drive run directory supplies a checkpoint.
- Search the original Drive directory for `phase2/step3_ch_role_lifetime_500ep_seed5499`. Absence means missing, not failed.

### Gate A

- Every reused artifact has an embedded optimizer seed and matching hash.
- Duplicate download names such as `(1)` are resolved through metadata, never filename order.
- No incomplete lineage enters candidate selection.

## 4. Phase B — Step 3 observation schema v3

### Implementation

Add a new immutable schema, tentatively `step3_ch_context_v3`, without changing v2 checkpoints. Broadcast the following current-round observable values to every branch:

1. scheduled-CH residual-energy fraction;
2. scheduled-CH next-harvest forecast mean;
3. scheduled-CH forecast uncertainty;
4. normalized CH-to-BS distance;
5. normalized current target-cluster feasible backlog;
6. normalized scheduled-CH forwarding exposure or member count;
7. scheduled-CH alive indicator.

Do not include node IDs, future realized harvest, future CH assignments, or any post-action quantity. Action-conditioned forwarding load remains expressible through the branch action level plus the shared CH context.

Update the observation layout, checkpoint schema/version, input dimension, normalization blocks, manifests, and audit reports. The exogenous schedule generator and CH assignments remain byte-identical.

### Tests and Gate B

- Observation shape/layout and finite-value tests.
- Same physical state plus different CH reserve produces different CH-context features.
- Permuting member rows and masks permutes branch Q values/actions inversely within the existing `1e-6` same-platform tolerance.
- Relabeling node IDs without changing physical state has no effect.
- Changing a non-observable future harvest sample has no effect.
- CH schedule hashes are identical before and after the schema change.

Stop if any identity, leakage, or schedule test fails.

## 5. Phase C — correctly wired reward and QoS gates

### Implementation

- Move contribution accounting into the active trainer term list so `ch_depletion_risk` is included automatically.
- Report physical reward, QoS penalty, CH-risk contribution, and their absolute fractions for every episode and final window.
- Add a greedy target-QoS evaluation at each stability checkpoint. Do not use exploratory training rows alone for checkpoint acceptance.
- Add a final-window QoS gate using the frozen repaired contract:
  - target delivery ratio `>= 0.55`;
  - target stale ratio `<= 0.45`;
  - target service fairness `>= 0.70`;
  - joint pass on at least 90% of greedy curriculum pairs in each of the last three stability snapshots.
- Keep the CH-risk absolute reward fraction below 20%; require it to be nonzero in every development seed.
- Do not change thresholds after observing a full-run result.

### QoS-controller calibration

Do not arbitrarily multiply the constraint. Run a development-only reward-geometry calibration that selects a penalty normalization before full training. When a delivery violation is active, the QoS penalty should occupy a predeclared 2–10% of total absolute learning reward; below 1% is treated as inert and above 20% as dominating. Compare:

- current episode-end dual update;
- an EMA-smoothed episode violation update;
- a nonzero delivery-multiplier floor calibrated only from development rollouts.

The controller variant is eligible only if it improves greedy target-QoS feasibility without increasing stale drops or disabling the CH-risk signal.

### Gate C

- Unit tests reproduce the recovered seed-6499 false-pass and prove the repaired gate rejects it.
- Synthetic tests prove risk inclusion and QoS penalty-fraction calculations.
- No full training until a short probe passes QoS activation, CH-risk activation, non-domination, and permutation audits.

## 6. Phase D — CH-risk strength ablation

The recovered risk fraction was only about 0.36% on average, so weight `1` may be too weak even though it is safe. After schema v3 makes the CH condition observable, evaluate frozen candidate weights `{1, 5, 10, 20}` using the same development schedules and optimizer seed in a short screening run.

Use successive halving to conserve compute:

1. 25-episode wiring smoke: reject nonfinite, inactive, identity-breaking, or dominating variants.
2. 100-episode screening: reject variants failing greedy target QoS or showing no CH-conditioned action response.
3. 250-episode comparison for at most two surviving variants.

Selection order is: QoS feasible first, then paired FND improvement, then throughput and packets/J. Never select the least-bad infeasible variant. These screening runs are hyperparameter development and cannot be reported as confirmation evidence.

## 7. Phase E — failure-safe training and finalization

### Trainer changes

- Save `training_complete_weights.pt` atomically immediately after episode 500 and before final greedy evaluation.
- Save `episodes.jsonl`, QoS-controller state, optimizer state, RNG states, curriculum order, and stability metadata before finalization.
- Run final greedy evaluation as a separate resumable command consuming the frozen episode-500 checkpoint.
- Add an idempotent finalizer capable of reconstructing the summary from the persisted checkpoint, episode log, and stability snapshots.
- Label model-only snapshots as salvage checkpoints, not exact resumptions, unless replay state and all RNG state are also serialized and verified.

### Notebook changes

- Synchronize episode logs and each stability snapshot to Drive as soon as it appears.
- At episode 500, copy the training-complete checkpoint to Drive before starting evaluation.
- Resume completed stages by manifest and hash, not simply by file existence.
- Print and persist subprocess return code, final 200 log lines, runtime fingerprint, and stop reason.
- Train only missing or invalid lineages; never rerun seed 6499 automatically.

### Gate E

Inject a deliberate interruption immediately after episode-complete checkpoint creation. A fresh runtime must finalize the lineage without retraining and reproduce the same checkpoint hash and summary metrics.

## 8. Phase F — revised full training

Because schema v3 changes the model input and reward/controller geometry, seed 6499 cannot initialize or count as a revised-model lineage. Retain it only as a diagnostic comparator.

After Phases B–E freeze, use three fresh optimizer seeds, proposed `5599, 6599, 7599`, for 500 episodes at the 1,200-round horizon. Development environment seeds remain `2400–2404`; registered seeds `3100–3104` and reserved confirmation seeds `3400–3404` remain untouched.

Every lineage must pass:

- complete curriculum and finite-value gates;
- same-platform permutation foundation;
- convergence and stability;
- greedy target-QoS final-window gate;
- CH-risk activation and non-domination;
- checkpoint/finalization integrity.

One failed lineage does not authorize threshold relaxation. Record the failure and stop selection if fewer than the predeclared number of valid lineages complete.

## 9. Phase G — paired 3,000-round development evaluation

Evaluate each structurally accepted checkpoint and all baselines on identical schedules for seeds `2400–2404`. Required outputs:

- whole-network delivery, stale-drop ratio, service fairness, throughput, energy, and packets/J;
- FND/HND with censor flags and Kaplan–Meier summaries where estimable;
- dying-node identity and overlap with energy-proportional;
- whether the dying node is the scheduled CH at FND;
- member-TX versus CH-RX, aggregation, and CH-to-BS energy;
- allocation/action distinctness on common states;
- paired per-seed deltas against the original seed-7399 checkpoint and energy-proportional.

Development candidate eligibility requires all five global QoS trials to pass, no structural failure, and nonnegative median paired FND change versus the original Step-2 candidate. Prefer a candidate that is not worse on FND in at least four of five development seeds. With only five pairs, do not claim conventional statistical significance.

## 10. Phase H — freeze and confirmation

Only after code, schema, risk weight, QoS controller, thresholds, baselines, endpoints, and analysis scripts are frozen may seeds `3400–3404` be opened. Report effect sizes and paired raw trials. For a publication-level significance claim, preregister a larger independent seed set; five paired trials alone cannot yield a two-sided exact Wilcoxon result below `0.0625`.

B16 remains a secondary idle-listening-disabled study. Any death or lifetime conclusion must be stated as B16-specific and cannot establish the primary idle-on hybrid-harvest contribution.

## 11. Ordered implementation checklist

1. Freeze the recovery manifest and seed-6499 diagnostic checkpoint.
2. Implement and test `step3_ch_context_v3`.
3. Repair reward-balance and greedy target-QoS gating.
4. Implement QoS reward-geometry calibration.
5. Implement atomic pre-evaluation checkpointing and resumable finalization.
6. Run interruption/recovery and same-platform foundation tests.
7. Run the risk-weight/controller successive-halving screen.
8. Freeze one configuration and train fresh seeds `5599/6599/7599`.
9. Run the paired 3,000-round development evaluation.
10. Select only a fully feasible candidate or record `no_candidate_global_qos_feasible`.
11. Freeze the entire evaluation contract, then open confirmation seeds.

## 12. Immediate stop conditions

- CH context is not demonstrably observable without identity leakage.
- Any permutation/foundation gate fails.
- Greedy target-QoS joint pass is below 90% in a required stability window.
- CH risk is inactive or exceeds 20% of absolute reward.
- A checkpoint cannot be recovered after injected interruption.
- A candidate fails any of the five global development QoS trials.
- Fewer than the required valid fresh lineages complete.

In every case, preserve the failure artifact and do not weaken the gate post hoc.
