# HTA-MAC QoS-repaired execution and improvement handoff

> **Mandatory preflight addendum:** read
> `HTA_MAC_EXTERNAL_REVIEW_RESOLUTION_AND_PREFLIGHT_DECISION_20260808.md`
> before executing the Colab notebook. It establishes that the active model is
> `EquivariantSetBranchingC51`, re-passes the current-code permutation gate,
> marks B16 as a secondary side study rather than the primary C1/C3 track, and
> reports measured budget utilization/contention. Where this handoff's earlier
> wording is less specific, the preflight addendum controls.

**Architecture decision:** equivariant set architecture authorized; flattened
`GlobalBranchingDuelingC51` remains retired.  
**Track decision:** paper-aligned B16 is a secondary literature side study.

**Prepared:** 8 August 2026  
**Repository:** `F:\WSN\matlab\stage2\hta-mac`  
**Upstream frozen assets:** `F:\WSN\matlab\stage2\final_repo`  
**Current status:** implementation and local validation complete; fresh GPU training has not yet been executed  
**Primary next input:** the trained-results ZIP produced by the supplied Colab notebook

---

## 1. Purpose of this handoff

This document gives the next agent enough technical and experimental context to:

1. run or supervise the repaired HTA-MAC training correctly;
2. analyze the resulting checkpoints and metrics without repeating earlier interpretation errors;
3. compare the result with relevant papers under defensible claim boundaries;
4. diagnose failed gates or weak performance;
5. propose and implement the next improvement as a new, preregistered development experiment; and
6. preserve all old evidence and untouched seed sets.

The most important rule is: **never fabricate an improvement and never weaken a frozen gate after seeing the result**. A failed experiment is valid evidence and must be preserved.

---

## 2. Scientific scope and non-negotiable boundaries

### 2.1 Learned intervention

The learned contribution is **HTA-MAC only**: adaptive intra-cluster MAC slot allocation.

- Cluster-head selection is exogenous and frozen.
- Do not retrain HEART-CH.
- Do not add routing, Pointer Networks, or a second learned CH policy.
- Do not combine results from HERMES with HEART-CH/HTA-MAC; their simulators and accounting are incompatible.

### 2.2 Paper-aligned branch

The repaired experiment uses a development-only paper-aligned profile:

- 100 static nodes;
- 100 m × 100 m field;
- base station at `(50, 50)`;
- 0.5 J initial node energy;
- 20% exogenous balanced rotating cluster heads;
- 4,000-bit data packets and 500-bit control packets;
- trained solar HMM;
- thermal harvesting disabled;
- idle listening disabled to match the closest published omissions;
- frame-slot budget `B = 16`;
- per-node cap `n_max = 3`;
- queue capacity 5 packets;
- packet TTL 3 rounds.

This is **paper-aligned**, not a reproduction of any third-party implementation, unpublished trace, or numerical experiment.

### 2.3 Seed firewall

| Purpose | Seeds | Rule |
|---|---:|---|
| Fresh repaired optimizer lineages | 5399, 6399, 7399 | May be used for training only |
| Development schedules | 2400–2404 | May be used for calibration, selection, diagnostics, and development evaluation |
| Reserved confirmation | 3400–3404 | Must remain unused until code, checkpoint, thresholds, endpoints, and analysis are frozen |
| Registered historical held-out set | 3100–3104 | Prohibited in this paper-aligned branch |
| Superseded pre-repair lineages | 5299, 6299, 7299 | Preserve as old evidence; never resume or relabel as repaired models |

---

## 3. Evidence available before the repair

The trained pre-repair folder is:

`HTA_MAC_PaperAligned_B16_Trained_Results_20260806/`

Across five development seeds and 300 rounds, the selected seed-5299 checkpoint produced approximately:

- median throughput: 29,980 packets;
- median generated packets: 30,100;
- global end-to-end delivery ratio: 0.996013;
- median stale-drop ratio: 0;
- global cumulative-service fairness: 0.999998;
- all 100 nodes alive at round 300;
- no observed FND or HND event for any policy.

Relative to internal comparators:

- versus static equal TDMA: +25.09% throughput, +22.04% energy consumption, and +2.60% packets/J;
- versus energy-proportional: essentially tied in throughput and slightly worse in packets/J;
- versus random-budgeted: +34 median packets but slightly worse energy efficiency;
- versus S2A2MAC-adapted: +29.62% throughput, +34.64% energy use, and −3.67% packets/J.

These results show near-ceiling short-horizon service, but they do **not** establish lifetime superiority, energy-efficiency superiority, or statistical superiority. With five nonzero paired differences, the smallest possible two-sided exact Wilcoxon p-value is 0.0625.

---

## 4. Critical defect discovered in the old QoS objective

### 4.1 What happened

The dynamic training wrapper changes the target cluster as the exogenous schedule rotates. The old QoS controller accumulated:

- packets delivered from the current target members' pre-existing queues; and
- packets newly generated only while those members were in the current target.

A packet could be generated while its node was outside the target and delivered later when that node entered the target. Consequently, numerator and denominator did not describe the same traffic cohort.

One seed-5299 episode reported:

- delivered: 2,806 packets;
- generated: 1,192 packets.

The controller silently clipped the ratio to 1.0. Therefore, the old training log is not valid evidence of an end-to-end delivery constraint, even though the independent Phase 3 global delivery calculation remains valid.

### 4.2 Fairness scope was also conflated

- Phase 2 target-cluster cumulative-service fairness was approximately 0.703–0.708.
- Phase 3 global network cumulative-service fairness was approximately 0.999998.

Both can be meaningful, but they are different metrics and must have different names and gates.

---

## 5. What was implemented

### 5.1 Cohort-consistent training metric

The training environments now emit:

```text
target_packets_offered = sum(pre-service queued packets for alive members of the current target cluster)
```

Packets delivered in that decision are an exact subset of this same-step offered backlog. The repaired controller uses:

```text
ratio_scope = episode_cumulative_target_backlog_service
demand_field = target_packets_offered
fairness_metric_name = target_cluster_service_fairness
```

Important interpretation:

- this is a **target-backlog service-opportunity ratio**;
- it is not global end-to-end generated-packet delivery;
- the repaired controller rejects `delivered > offered`;
- repaired ratios are not silently clipped;
- legacy schema-2 configurations retain their historical clipping behavior so old runs remain reproducible.

### 5.2 Separate whole-network evaluation contract

Phase 3 continues to calculate the independent global metrics:

- `global_network_end_to_end_delivery_ratio`;
- `global_network_stale_drop_ratio`;
- `global_network_service_fairness`;
- energy consumed and packets/J;
- FND and HND event indicators;
- censor round and right-censor status;
- Kaplan–Meier median when estimable;
- restricted mean event-free rounds at a common horizon.

The frozen development gates are:

- global delivery ratio ≥ 0.95;
- global stale-drop ratio ≤ 0.01;
- global service fairness ≥ 0.95.

These global gates are deliberately separate from the target-service training constraint.

### 5.3 Fresh training and selection sequence

The notebook performs:

1. bundle SHA-256 verification;
2. safe ZIP extraction with path traversal rejection;
3. verification of every file against the embedded manifest;
4. dependency installation, compilation, and the complete validation suite;
5. development-only C51 return-scale recalibration under the repaired controller;
6. three fresh 500-episode lineages using seeds 5399/6399/7399;
7. curriculum, convergence, stability, projection, permutation, and C51-support audits;
8. 300-round whole-network paired development evaluation;
9. deterministic lexicographic development selection;
10. common-state policy-action distinctness audit;
11. 3,000-round paired development evaluation with censor-aware lifetime reporting;
12. evidence packaging and SHA-256 sidecar generation.

### 5.4 Action-distinctness audit

Every comparator is queried on exactly the same state before the environment advances using HTA-MAC's action. The audit reports:

- exact round-action agreement fraction;
- mean normalized L1 action distance;
- active-node-set Jaccard similarity;
- mean slots used;
- full trajectory action signatures.

This answers whether HTA-MAC actually makes different decisions from energy-proportional, static, FFSS-adapted, S2A2MAC-adapted, harvest-proportional, or random-budgeted policies. It is not an outcome comparison.

A two-round smoke run using the old seed-5299 checkpoint completed successfully. It showed 0% exact agreement with energy-proportional in that tiny diagnostic, confirming that the audit can distinguish policies. This smoke result is not scientific evidence and was not retained as an experimental artifact.

---

## 6. Files added or modified

### Core implementation

- `agents/qos_constraints.py`
  - supports legacy and repaired metric contracts;
  - adds `evaluate_info`;
  - stores metric-contract metadata;
  - rejects invalid repaired cohorts;
  - preserves legacy state/checkpoint compatibility.
- `envs/dynamic_cluster_training_env.py`
  - emits target offered backlog and explicitly named target fairness.
- `envs/fixed_cluster_training_env.py`
  - emits the same repaired fields for consistent tests and calibration.
- `experiments/train_phase2_dynamic_curriculum.py`
  - delegates QoS extraction to the frozen controller contract.
- `experiments/calibrate_paper_aligned_return_scale.py`
  - calibrates through the same repaired controller path.
- `experiments/audit_policy_action_distinctness.py`
  - performs the common-state decision audit.

### Frozen repaired configurations

- `config/paper_aligned_hasani2025_b16_qos_repaired.json`
- `config/paper_aligned_hasani2025_qos_constraints_repaired.json`
- `config/paper_aligned_hasani2025_qos_feasibility_repaired.json`
- `config/paper_aligned_hasani2025_global_evaluation_gates_repaired.json`

### Validation

- `validation/test_qos_repaired_contract.py`

The complete repository suite passed:

```text
92 passed, 181 warnings
```

The warnings were dependency deprecations from PyTorch Geometric/Torch and not test failures.

### Build and user artifacts

- `tools/build_paper_aligned_b16_qos_repaired_bundle.ps1`
- `tools/generate_qos_repaired_colab.py`
- `colab/HTA_MAC_PaperAligned_B16_QoSRepaired_Training_Colab_20260808.ipynb`
- `colab/HTA_MAC_PaperAligned_B16_QoSRepaired_Training_Bundle_20260808.zip`
- `colab/HTA_MAC_PaperAligned_B16_QoSRepaired_Training_Bundle_20260808.zip.sha256`
- `HTA_MAC_PAPER_COMPARISON_AND_QOS_REPAIR_IMPLEMENTATION_20260808.md`
- `HTA_MAC_TRANSFER_MANIFEST_20260808.json` (generated after the final bundle and notebook)

Final bundle SHA-256:

```text
Use `colab/HTA_MAC_PaperAligned_B16_QoSRepaired_Training_Bundle_20260808.zip.sha256`; the supplied notebook locks the same value.
```

Final notebook SHA-256 at handoff time:

```text
Compute from the supplied notebook at transfer time; it is not an input to training.
```

The bundle contains 184 manifest-verified files and includes the frozen shared-contract checkpoint required by the full Colab validation suite.

**Intended transfer mechanism:** send the report, mandatory preflight addendum, bundle, `.sha256` sidecar, notebook, and external transfer manifest together. The report is not intended to be the standalone integrity anchor because embedding the bundle hash in a report that is itself bundled creates a circular hash dependency. `HTA_MAC_TRANSFER_MANIFEST_20260808.json` is the external machine-readable anchor containing the actual hashes.

---

## 7. Challenges encountered and how they were resolved

### Challenge 1: inconsistent Windows ZIP separators

Earlier bundles could contain backslashes such as `stage2\COLAB_...json`, while Colab expects POSIX paths.

**Resolution:** the notebook normalizes every member name with backslash-to-slash conversion, rejects absolute/parent-traversal paths, and extracts each member explicitly.

### Challenge 2: validation expected a frozen checkpoint absent from the bundle

The test `test_frozen_hta_checkpoint_obeys_shared_contract` failed when `outputs/phase2/authoritative_dynamic_budget8_500ep/branching_c51.pt` was missing.

**Resolution:** the bundle builder now requires and includes both that checkpoint and its summary. The build aborts if either file is absent.

### Challenge 3: training failure message hid the original process failure

An earlier notebook asserted that the output directory existed, producing `No output produced for seed ...` without making the process exit and log location sufficiently clear.

**Resolution:** the repaired notebook streams stdout/stderr to a Drive log, records the return code, checks for the exact run directory, and reports both exit status and log path.

### Challenge 4: invalid QoS cohort and silent clipping

This was the central scientific defect.

**Resolution:** use same-step offered backlog for the training service ratio, enforce subset invariants, store explicit metric semantics, and keep global delivery separate.

### Challenge 5: short-horizon lifetime ceiling

At 300 rounds every policy retained all nodes, so FND/HND could not discriminate models.

**Resolution:** add a 3,000-round development evaluation and censor-aware analysis. The horizon is never substituted for an unobserved event.

### Challenge 6: outcome tie did not reveal whether policies were identical

HTA-MAC and energy-proportional had nearly identical throughput, but outcome similarity alone cannot show whether they choose the same actions.

**Resolution:** add the common-state action-distinctness audit before deciding what model improvement is needed.

### Challenge 7: weak git provenance inside Colab archives

Earlier Colab logs reported that the extracted bundle was not a git repository.

**Resolution:** the manifest records the Stage 2 source commit, HTA-MAC working-tree status, frozen HEART-CH commit, configuration hashes, and every bundled file hash. Use the manifest rather than expecting `.git` inside Colab.

### Challenge 8: Firecrawl research index was unauthenticated

The local Firecrawl CLI was installed but not authenticated.

**Resolution:** literature claims were verified through primary publisher pages and author manuscripts. Do not claim that Firecrawl retrieval succeeded.

---

## 8. Literature comparison boundary

Relevant primary sources include:

- Hasani et al. (2025): [Scientific Reports](https://www.nature.com/articles/s41598-025-14111-y)
- Ge, Nan, and Guo (2021): [International Journal of Distributed Sensor Networks](https://journals.sagepub.com/doi/10.1177/15501477211007411)
- Eris, Gul, and Boluk (2024): [Sensors / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11487392/)
- HENO-MAC: [arXiv](https://arxiv.org/abs/2401.00717)
- FFSS/AFSS: [IET publisher page](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/cmu2.12243)
- SHR-TDMA: [IET publisher page](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/iet-com.2019.0977)
- Structure-aware RL: [arXiv](https://arxiv.org/abs/1807.08315)

Defensible positioning:

- HTA-MAC's +25.09% throughput over its own static baseline is numerically larger than Hasani et al.'s reported 11.79% and lies inside Ge et al.'s reported 16.6%–30.1% range.
- This does not prove cross-paper superiority because simulator, traffic, harvesting, action space, horizon, and baselines differ.
- HTA-MAC currently has stronger internal feasibility and audit evidence than a throughput-only presentation.
- HTA-MAC currently lacks HENO-MAC-like real-trace/priority-delay evidence and Eris et al.-style observed lifetime separation.

Never compare absolute FND/HND values with the underwater paper. Its acoustic energy model is incompatible with this terrestrial radio environment.

---

## 9. How to run the repaired experiment

Upload this ZIP to Colab:

```text
/content/HTA_MAC_PaperAligned_B16_QoSRepaired_Training_Bundle_20260808.zip
```

Open and run:

```text
HTA_MAC_PaperAligned_B16_QoSRepaired_Training_Colab_20260808.ipynb
```

Frozen notebook settings:

```python
BUNDLE_PATH = '/content/HTA_MAC_PaperAligned_B16_QoSRepaired_Training_Bundle_20260808.zip'
DRIVE_OUTPUT_DIR = '/content/drive/MyDrive/HTA_MAC_PaperAligned_B16_QoSRepaired_20260808'
SEEDS = [5399, 6399, 7399]
EPISODES = 500
# EXPECTED_BUNDLE_SHA256 is already frozen inside the supplied notebook.
```

Expected NVIDIA L4 time is roughly 4–9 hours. Actual duration depends on Colab load and the long-horizon evaluation. Each completed lineage is synchronized to Drive, allowing safe continuation after an interruption.

Expected final results archive:

```text
HTA_MAC_PaperAligned_B16_QoSRepaired_Trained_Results_20260808.zip
```

---

## 10. Mandatory analysis procedure when trained results arrive

### Step 1: integrity and provenance

1. Hash the received results ZIP.
2. Extract safely without overwriting old results.
3. Locate and parse the embedded Colab manifest.
4. Confirm optimizer seeds are exactly 5399/6399/7399.
5. Confirm development seeds are exactly 2400–2404.
6. Search all JSON, CSV, and logs for forbidden seeds 3100–3104 and reserved seeds 3400–3404.
7. Verify the repaired QoS config hash and `demand_field=target_packets_offered`.
8. Record any runtime, dependency, CUDA, or git-provenance warnings.

Stop immediately if a checksum differs, a forbidden seed appears, or a lineage reused an old checkpoint.

### Step 2: Phase 2 lineage gates

For every optimizer seed, inspect:

- episodes requested/completed;
- `phase2_curriculum_gate_pass`;
- convergence evidence;
- stability snapshots;
- reward-balance domination;
- nonfinite detection;
- always-sleep collapse;
- C51 return/support saturation;
- foundation audit status;
- permutation equivariance;
- action budget/cap feasibility;
- final QoS multipliers and violations;
- cumulative delivered/offered/stale counts.

Explicitly verify:

```text
cumulative delivered <= cumulative offered
```

for the repaired controller. Do not accept clipping as a substitute.

### Step 3: candidate selection

Recompute selection independently from raw Phase 3 CSV files. Confirm each candidate's five trials satisfy or fail:

- delivery ≥ 0.95;
- stale ratio ≤ 0.01;
- global fairness ≥ 0.95.

Reproduce the lexicographic ordering:

1. joint global QoS pass count descending;
2. median constraint violation ascending;
3. median throughput descending;
4. median packets/J descending;
5. optimizer seed ascending as deterministic tie-break only.

If no candidate passes all five schedules, retain `no_candidate_global_qos_feasible`; do not select the least-bad candidate.

### Step 4: 300-round policy comparison

For each policy, report median and IQR for:

- throughput;
- generated packets;
- delivery ratio;
- stale-drop ratio;
- energy consumed;
- packets/J;
- global service fairness;
- residual-energy fairness and coefficient of variation;
- mean/minimum residual energy;
- FND/HND event counts and censoring.

Calculate paired per-seed differences against static, energy-proportional, S2A2MAC-adapted, FFSS-adapted, harvest-proportional, and random-budgeted policies. Treat p-values as descriptive development statistics only.

### Step 5: action-distinctness audit

Answer:

- Is HTA-MAC exactly duplicating any baseline?
- Is it selecting the same active nodes but changing slot multiplicities?
- Is its slot use systematically higher, explaining the energy cost?
- Does distinctness vary by schedule seed?
- Is the learned action trajectory deterministic for the frozen checkpoint?

If HTA-MAC is close to energy-proportional in outcomes but distant in actions, inspect whether the environment is at a throughput ceiling. If it is close in both outcomes and actions, the learned model may have collapsed toward that heuristic.

### Step 6: 3,000-round lifetime analysis

For FND and HND:

- count observed events and censored trials per policy;
- report Kaplan–Meier median only if survival crosses 0.5;
- report restricted mean event-free rounds at the common restriction horizon;
- never encode unobserved events as 3,000;
- compare energy and packets/J jointly with survival.

If no FND occurs by 3,000 rounds, the result remains censored. Propose a longer **development-only** horizon before confirmation, justified from energy-depletion trajectories, and freeze it before running.

---

## 11. Performance-diagnosis decision tree

### Case A: repaired QoS training fails

Inspect, in order:

1. offered-backlog zero frequency;
2. delivered/offered and stale/offered trajectories;
3. fairness warm-up behavior;
4. multiplier growth and saturation at the configured maximum;
5. raw reward versus constraint penalty magnitude;
6. reward-balance domination;
7. queue-feasible caps and budget projection;
8. C51 support occupancy and top-atom saturation;
9. convergence and stability windows.

Do not lower a threshold after seeing these lineages. Create a new config version, new optimizer seeds, and a written rationale if the objective itself must change.

### Case B: throughput is high but energy efficiency remains weak

Likely causes:

- allocating multiple slots when one would preserve delivery;
- service reward dominating energy cost;
- operating at a throughput ceiling where extra transmissions add little value;
- insufficient state representation of marginal transmission energy;
- no explicit energy-efficiency or residual-energy risk constraint.

Candidate next experiment, only after analyzing the fresh result:

1. add marginal joules-per-successful-packet or energy-normalized advantage to the observation/reward audit;
2. add an allocation-sparsity or marginal-slot-cost regularizer;
3. retain the hard B16 projection;
4. recalibrate C51 scale;
5. use entirely new optimizer seeds;
6. require non-inferior global delivery/stale/fairness before accepting an energy gain.

Do not tune directly on confirmation seeds.

### Case C: HTA-MAC ties energy-proportional

Use the action audit:

- **Different actions, same outcomes:** likely environment/metric ceiling. Emphasize longer lifetime, burst traffic, stronger energy scarcity, or delay—not a larger network solely to manufacture a difference.
- **Similar actions and outcomes:** inspect feature attribution/ablations for energy, queue, harvest forecast, TTL/cap, and trajectory blocks. The learned policy may have converged to an energy heuristic.

Any stress environment must be frozen using development evidence and must remain scientifically plausible. Report it as a stress test, not replacement of the main environment.

### Case D: fairness is low

First confirm whether the low value is target-cluster fairness or global fairness. Then inspect:

- membership rotation and opportunity counts;
- service debt per node;
- whether cumulative service/history is visible in the policy observation;
- whether fairness is computed only over eligible nodes or over the full network;
- multiplier warm-up and update interval.

A potential improvement is an explicit service-debt feature or virtual queue, but it changes the observation contract and requires new architecture tests, scale calibration, seeds, and checkpoints.

### Case E: lifetime events remain unobserved

Do not claim success from “all nodes alive.” Use residual-energy slope and minimum-energy trajectories to choose a longer development horizon. Prefer censor-aware survival endpoints. If extending beyond 3,000 rounds is computationally expensive, first estimate the plausible event window from the five development energy trajectories; then freeze one new horizon.

### Case F: a lineage fails only a structural audit

Discard it from candidate selection. Do not average it with accepted lineages. Diagnose the specific architecture/projection/C51 failure and rerun as a new experiment version if necessary.

---

## 12. Recommended improvement priorities after the fresh run

Do not implement all of these blindly. Choose based on the observed failure mode.

1. **Energy-aware marginal allocation:** reduce unnecessary second/third slots while preserving ≥0.95 global delivery.
2. **Service-debt state:** improve target fairness without confusing it with global fairness.
3. **Monotonicity audit/regularizer:** test whether allocations respond sensibly to higher queue, higher energy, lower transmission cost, and imminent TTL expiration.
4. **Delay/age endpoints:** add mean/percentile packet age and deadline miss rate if the simulator can validate them exactly.
5. **Real solar traces:** introduce only with documented provenance and a separate experiment profile; do not relabel the trained Stage 1 HMM as a real trace.
6. **Longer development survival window:** only if 3,000 rounds remain censored.
7. **Confirmation run:** last, after one checkpoint and all analysis choices are frozen.

Acceptance for any proposed improvement should require:

- no regression in delivery, stale drops, or global fairness;
- improved median packets/J or censor-aware lifetime on development seeds;
- action distinctness from trivial heuristics;
- all structural and numerical gates passing;
- no use of confirmation or prohibited seeds.

---

## 13. Stop conditions

Stop and preserve evidence if any of the following occurs:

- bundle or manifest checksum mismatch;
- unsafe ZIP member;
- complete validation suite failure;
- missing authoritative checkpoint required by validation;
- forbidden or reserved seed usage;
- training output directory absent after a nonzero process exit;
- nonfinite rewards, losses, Q-values, or metrics;
- C51 support saturation gate failure;
- action budget/cap violation;
- permutation/identity audit failure;
- pathological reward domination;
- no lineage passing every structural gate;
- no candidate passing all five global development trials;
- failed action audit or long-horizon evaluation.

Do not bypass a stop by modifying the notebook in place. Diagnose locally, version the protocol/configuration, rebuild the bundle, recalculate its checksum, and start fresh optimizer seeds.

---

## 14. Useful local verification commands

From `F:\WSN\matlab\stage2\hta-mac`:

```powershell
python -B -m pytest validation -q -p no:cacheprovider
```

```powershell
python -m py_compile agents/qos_constraints.py envs/dynamic_cluster_training_env.py envs/fixed_cluster_training_env.py experiments/train_phase2_dynamic_curriculum.py experiments/calibrate_paper_aligned_return_scale.py experiments/audit_policy_action_distinctness.py
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_paper_aligned_b16_qos_repaired_bundle.ps1
```

Do not rebuild the ZIP unless code/configuration changes. Rebuilding changes the ZIP checksum and requires regenerating the notebook with the new checksum.

---

## 15. Required final report after training

The post-training report should contain:

1. executive conclusion;
2. archive integrity and provenance;
3. exact environment/profile/configuration;
4. Phase 2 results for every lineage;
5. repaired QoS cohort invariant evidence;
6. structural audit results;
7. independently reproduced development selection;
8. 300-round paired metrics and energy trade-offs;
9. action-distinctness findings;
10. 3,000-round censor-aware lifetime results;
11. comparison with primary papers;
12. what HTA-MAC is better at, tied at, worse at, and still unknown;
13. root-cause analysis for every failed or weak metric;
14. one prioritized next experiment with frozen gates and fresh seeds;
15. explicit confirmation-readiness decision.

Every numerical claim must point to an artifact path and field/column. Separate measured result, inference, hypothesis, and proposed work.

---

## 16. Ready-to-paste prompt for the next agent

```text
Work in F:\WSN\matlab\stage2\hta-mac. Read HTA_MAC_QOS_REPAIRED_AGENT_EXECUTION_HANDOFF_20260808.md completely before acting. The user will provide HTA_MAC_PaperAligned_B16_QoSRepaired_Trained_Results_20260808.zip or its extracted folder.

First perform a read-only integrity and provenance audit: verify the ZIP hash, embedded manifest, file hashes, repaired QoS configuration, optimizer seeds 5399/6399/7399, development seeds 2400-2404, and confirm that reserved 3400-3404 and prohibited 3100-3104 were not used. Do not overwrite or delete the old pre-repair results.

Then parse every Phase 2 summary/log/audit and independently verify the repaired cohort invariant delivered <= offered, all structural gates, C51 support, convergence, stability, reward balance, and QoS multiplier behavior. Recompute the 300-round development selection from raw CSV files using the frozen global gates. Analyze the common-state action-distinctness audit and the 3,000-round censor-aware lifetime evaluation. Never convert a censored FND/HND into the horizon.

Compare measured HTA-MAC performance with static, energy-proportional, harvest-proportional, S2A2MAC-adapted, FFSS-adapted, and random-budgeted policies. Then compare contextually with the primary papers linked in the handoff. Do not claim cross-paper reproduction or superiority.
In the required final report, explicitly state that the zero-firing death-penalty finding is specific to the idle-listening-disabled B16 side profile. Do not generalize it to the primary idle-on hybrid-harvest track without measuring that track independently.


Create a detailed Markdown report containing evidence paths, exact metrics, what passed, what failed, what HTA-MAC is better/worse/tied at, root causes, and one prioritized next development experiment. Do not implement a new optimization until the trained evidence identifies the failure mode. If implementation is justified, version all configs, use fresh optimizer seeds, preserve the seed firewall, add regression tests, run the complete validation suite, and build a new checksum-locked Colab notebook and ZIP. Never weaken a gate after seeing results and never fabricate improvements.
```

---

## 17. Final status at handoff

| Item | Status |
|---|---|
| QoS cohort defect | Repaired in code |
| Metric scope naming | Repaired |
| Regression tests | Passing |
| Full validation | 87 passed |
| Action-distinctness audit | Implemented and smoke-tested |
| 3,000-round development evaluation | Implemented in notebook |
| Fresh repaired GPU training | Pending user Colab execution |
| Fresh measured improvements | Not yet available |
| Confirmation seeds | Untouched |
| Publication/lifetime/superiority claim | Not authorized by current evidence |

The next correct action is to run the supplied Colab notebook or analyze its completed results archive. No further performance tuning should occur until that evidence is available.
