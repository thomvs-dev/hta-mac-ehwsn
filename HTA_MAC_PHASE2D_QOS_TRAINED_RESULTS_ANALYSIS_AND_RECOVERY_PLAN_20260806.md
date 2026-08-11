# HTA-MAC Phase 2D QoS Trained Results: Implementation Audit, Results, Failures, and Recovery Plan

**Prepared:** 6 August 2026  
**Repository:** `F:\WSN\matlab\stage2\hta-mac`  
**Input archive:** `HTA_MAC_Phase2D_QoS_Trained_Results_20260804.zip`  
**Archive SHA-256:** `A4F2FEEADB75953BE5356A66033538E85FB8A6686EA9B8E6D8372E69C9EAA24E`
**Sidecar verification:** `HTA_MAC_Phase2D_QoS_Trained_Results_20260804.zip.sha256` matches the recomputed archive hash.

## 1. Executive verdict

The training run is technically valid but the Phase 2D QoS research gate is **not passed**.

All three 500-episode lineages completed, covered the full 25-pair development curriculum, remained finite, avoided always-sleep collapse, passed convergence and local stability checks, and passed the equivariance/action-feasibility/C51-support audit. The archive-selected seed is `4299`, and its checkpoint hash is internally consistent.

However, the existing `phase2_curriculum_gate_pass=true` does not test absolute QoS satisfaction. It only proves that the constrained return stabilized and that several local metrics were stable. In the final 50 training episodes, all three QoS targets were simultaneously satisfied in only:

- seed 2299: 2% of episodes;
- seed 3299: 0% of episodes;
- seed 4299: 4% of episodes.

The selected seed-4299 checkpoint was then evaluated network-wide on development seeds `2300-2304`. It passed all three QoS thresholds on only 1/5 schedules. Its medians were delivery `0.53672` against a `0.55` minimum, stale-drop ratio `0.45128` against a `0.45` maximum, and fairness `0.97854` against a `0.95` minimum.

Therefore:

- the architecture and training pipeline are structurally sound;
- seed 4299 is the best current budget-12 development candidate;
- the model has learned a useful lifetime/throughput/energy trade-off;
- it has **not** met its frozen QoS contract at budget 12;
- held-out Phase 3 evaluation must remain locked until the development protocol is repaired.

## 2. Evidence and verification boundary

This report uses only:

1. the uploaded trained-results archive;
2. frozen development seeds `2300-2304`;
3. frozen horizon `300`;
4. the repository's existing Phase 3 network evaluator;
5. additional development-only snapshot and inference-budget diagnostics.

Held-out seeds `3100-3104` were not run or inspected. No publication-level superiority claim is made.

The archive contains 25 entries:

- `DEVELOPMENT_SELECTION.json`;
- three `summary.json` files;
- three 500-row `episodes.jsonl` histories;
- three final `branching_c51.pt` checkpoints;
- nine stability checkpoints at episodes 400, 450, and 500;
- three Phase 2D foundation audits.

The final checkpoint hashes recomputed from the ZIP exactly match both the selection record and foundation audits:

| Optimizer seed | Checkpoint SHA-256 | Integrity |
|---:|---|---|
| 2299 | `A8FF6323196A4ADA425681FC6A110BC9641D4AB03C511D28031066EEAC51CB5F` | Match |
| 3299 | `0D025AA0668C7D27D239031B421147B8B308D1791B7481B46F5237B764F9D00B` | Match |
| 4299 | `70B3682CE3AA1AD1888A4919AC526B603EC8D8CB0C26F2EDEABAA130EAAAD2D7` | Match |

One provenance limitation remains: `git_hash` is recorded as `unavailable` because the Colab bundle did not contain Git metadata. Configuration and checkpoint hashes are present, but a future results manifest must record the source commit explicitly.

## 3. What was implemented

### 3.1 Bounded system scope

The work remains MAC-only. HEART-CH cluster-head schedules and embeddings are frozen and replayed. No cluster-head retraining, routing redesign, Pointer Network, or HERMES result is mixed into Phase 2D.

### 3.2 Observation and architecture

The trained agent is an equivariant set Branching Dueling C51 network:

- state schema: `phase2d_ttl_cap_v2`;
- 58 input features per node;
- 18 physical features;
- 4 TTL/packet-age features;
- 4 action-validity/cap features;
- 32 frozen ST-GCN embedding features;
- embedding boundary at dimension 26;
- shared node encoder with masked set context;
- 4 actions per branch: 0, 1, 2, or 3 slots;
- projection budget: 12 slots;
- maximum padded branches: 100;
- online parameter count: 115,123.

Permutation-equivariance is exact in the supplied audits: all random and targeted permutations had zero reported log-probability/Q error, local argmax agreement `1.0`, projected allocation agreement `1.0`, and no budget/cap violation.

### 3.3 Distributional RL configuration

- C51 atoms: 51;
- support: `[-30, 30]`;
- discount: `0.99`;
- learning rate: `1e-5`;
- batch size: 32;
- replay capacity: 5,000;
- replay warm-up: 256;
- target update: every 250 learning steps;
- learning: every 4 environment steps;
- epsilon: `1.0 -> 0.05`;
- reward scale: `0.14436784678738615`;
- trajectory-order loss weight: `1.0`;
- concavity loss weight: `0.1`;
- precision: FP32.

All three categorical audits remained far from support saturation. Median boundary-atom mass was approximately `0.0275`, no Q value was within one atom of a support boundary, and all support gates passed.

### 3.4 QoS-constrained objective

The original six-term physical reward remains unchanged. A non-positive Lagrangian penalty is added to the raw learning reward before the frozen C51 scale. The frozen targets are:

- delivery ratio `>= 0.55`;
- stale-drop ratio `<= 0.45`;
- Jain queue-service fairness `>= 0.95`.

The controller updates every environment step from episode-cumulative target-cluster metrics. Initial multipliers are delivery `1.0`, stale `1.0`, and fairness `0.5`; learning rates are `0.05`, `0.05`, and `0.02`; all multipliers are capped at `10.0`. Multipliers persist across episodes, while episode counters reset.

### 3.5 Training and selection protocol

Each optimizer seed trained from scratch for 500 episodes across 25 frozen seed/rank curriculum cases. The notebook selected the lineage with minimum mean last-50 positive training QoS violation, using local greedy fairness and throughput as tie-breakers. That frozen rule selected seed `4299`:

| Rank | Seed | Last-50 positive violation | Local greedy throughput | Local greedy fairness |
|---:|---:|---:|---:|---:|
| 1 | 4299 | 0.136377 | 10,668.92 | 0.799912 |
| 2 | 3299 | 0.137651 | 10,747.64 | 0.793617 |
| 3 | 2299 | 0.144732 | 10,588.00 | 0.751252 |

This selection is procedurally reproducible, but Section 7 explains why the ranking metric is not sufficient for QoS promotion.

## 4. Training results

### 4.1 Completion and safety gates

| Seed | Episodes | Environment steps | Full curriculum | Finite | Collapse | Original reward domination | Convergence | Local stability | Foundation audit |
|---:|---:|---:|---|---|---|---|---|---|---|
| 2299 | 500 | 77,493 | Yes | Yes | No | No | Pass | Pass | Pass |
| 3299 | 500 | 78,196 | Yes | Yes | No | No | Pass | Pass | Pass |
| 4299 | 500 | 77,671 | Yes | Yes | No | No | Pass | Pass | Pass |

The original physical reward was not pathologically dominated by one of its six terms. Packets delivered contributed about 60-61% of absolute physical reward contribution in the final 50 episodes, below the configured 80% limit.

### 4.2 Last-50 constrained-objective behavior

| Seed | Raw physical reward | Constraint penalty | Constrained raw return | Delivery | Stale | Fairness | All-QoS pass rate | Negative constrained-return rate |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2299 | 268.97 | -357.80 | -88.83 | 0.54829 | 0.46603 | 0.85611 | 2% | 82% |
| 3299 | 274.63 | -371.22 | -96.59 | 0.54739 | 0.46668 | 0.86364 | 0% | 84% |
| 4299 | 280.70 | -338.55 | -57.85 | 0.54936 | 0.45840 | 0.86052 | 4% | 72% |

Important interpretation:

- seed 4299 is best among the three, but its training proxy still misses all three mean thresholds;
- the final-50 constraint penalty is larger in magnitude than the physical reward for every lineage;
- the system converged to a stable negative constrained return, not to constraint satisfaction;
- fairness multipliers were essentially saturated at `10.0` for 96-100% of the final 50 episodes;
- mean delivery multipliers were `9.66-9.70` and were also repeatedly saturated;
- stale multipliers fell to only `0.15-0.21`, even though network-wide stale QoS remained near or above its limit.

Thus, the dual controller was active but poorly calibrated to the actual network objective.

### 4.3 What “convergence pass” actually means

The convergence test compares the constrained raw return in episodes 401-450 against 451-500 and accepts relative change `<= 10%`:

| Seed | Previous-50 return | Last-50 return | Relative change | Gate |
|---:|---:|---:|---:|---|
| 2299 | -81.01 | -88.83 | 9.65% | Pass |
| 3299 | -94.15 | -96.59 | 2.60% | Pass |
| 4299 | -58.28 | -57.85 | 0.73% | Pass |

This is a valid stability result, but it is not a feasibility result. A stable infeasible plateau can pass this test.

## 5. Network-wide development evaluation

Because training used target-cluster proxies, all three final checkpoints were independently evaluated using the existing network-wide Phase 3 policy path on development seeds `2300-2304`, horizon 300, budget 12. Each evaluation completed 35/35 policy runs with no failure or non-finite metric.

### 5.1 Final checkpoint comparison

| Seed | Median FND | Throughput | Delivery | Stale | Fairness | Idle J | Packets/J | Joint QoS schedules |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2299 | 180 | 15,350 | 0.53168 | 0.45748 | 0.94412 | 30.0050 | 377.34 | 1/5 |
| 3299 | 213 | 15,354 | 0.52310 | 0.46501 | 0.93936 | 30.1062 | 376.66 | 0/5 |
| **4299** | **213** | **15,919** | **0.53672** | **0.45128** | **0.97854** | **31.1618** | **377.12** | **1/5** |

Seed 4299 is confirmed as the strongest balanced lineage. It satisfies fairness on 5/5 development schedules, stale QoS on 2/5, delivery QoS on 1/5, and all three together on 1/5.

Relative to its thresholds, the selected checkpoint's medians are:

- delivery shortfall: `0.01328` absolute;
- stale excess: `0.00128` absolute;
- fairness margin: `+0.02854` absolute.

The stale target is nearly reached, but delivery is not robustly reached.

### 5.2 Selected seed 4299 versus development references

| Policy | FND | Throughput | Delivery | Stale | Fairness | Idle J | Packets/J |
|---|---:|---:|---:|---:|---:|---:|---:|
| HTA-MAC seed 4299, B12 | 213 | 15,919 | 0.53672 | 0.45128 | 0.97854 | 31.1618 | 377.12 |
| S2A2MAC-adapted | 206 | 13,048 | 0.47589 | 0.51280 | 0.90502 | 31.7396 | 329.77 |
| Static equal | 131 | 12,268 | 0.66095 | 0.32104 | 0.98119 | 44.6596 | 232.38 |
| Energy proportional | 129 | 12,041 | 0.79495 | 0.19183 | 0.99121 | 44.9928 | 228.40 |
| Harvest proportional | 124 | 14,479 | 0.66951 | 0.32307 | 0.98644 | 43.8336 | 271.51 |
| FFSS-adapted | 143 | 12,049 | 0.65452 | 0.32739 | 0.98655 | 45.0424 | 227.59 |

On these development medians, seed 4299 improves all listed service, lifetime, and energy metrics over S2A2MAC-adapted. Against the service-oriented references it preserves much more energy and lifetime while sacrificing delivery and freshness. This is encouraging development evidence, but it must not be converted into a held-out or statistical superiority claim.

## 6. Snapshot and budget diagnostics

### 6.1 Episode 400, 450, and 500 network behavior

| Seed-4299 checkpoint | FND | Throughput | Delivery | Stale | Fairness | Idle J | Joint QoS schedules |
|---|---:|---:|---:|---:|---:|---:|---:|
| Episode 400 | 268 | 15,849 | 0.53261 | 0.45458 | 0.96324 | 31.1460 | 0/5 |
| Episode 450 | 261 | 15,851 | 0.53247 | 0.45480 | 0.96381 | 31.1338 | 0/5 |
| Episode 500 | 213 | 15,919 | 0.53672 | 0.45128 | 0.97854 | 31.1618 | 1/5 |

Late training improves delivery, stale ratio, and fairness slightly, but median FND falls from 268 to 213. The local stability gate reported only a 0.4% FND span, while the network-wide evaluator shows a 20.5% decline from episode 400 to 500. Therefore, the current local stability proxy is not a reliable network early-stopping criterion.

### 6.2 Inference-budget diagnostic

The frozen seed-4299 checkpoint was diagnostically projected at larger inference budgets. This is not a substitute for retraining at those budgets.

| Inference budget | FND | HND | Throughput | Delivery | Stale | Fairness | Idle J | Packets/J | Joint QoS schedules |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 213 | Censored at 300 | 15,919 | 0.53672 | 0.45128 | 0.97854 | 31.1618 | 377.12 | 1/5 |
| **16** | **169** | **263** | **16,891** | **0.64045** | **0.34775** | **0.98682** | **42.4104** | **315.33** | **5/5** |
| 20 | 132 | 205 | 14,709 | 0.68573 | 0.30429 | 0.98478 | 43.8160 | 275.43 | 5/5 |
| 24 | 114 | 151 | 13,005 | 0.73115 | 0.25620 | 0.98296 | 44.3384 | 246.98 | 5/5 |

Budget 16 is the first tested point that meets all QoS thresholds on all five development schedules. It retains better median FND, throughput, idle energy, and packets/J than static equal, but gives up much of budget 12's lifetime and energy advantage. This strongly indicates that the main remaining budget-12 QoS gap is capacity-limited, not only optimizer-limited.

Because the checkpoint was trained at budget 12, budget-16 inference is diagnostic only. A registered budget-16 branch must be retrained and revalidated before it can become the primary candidate.

## 7. What went wrong

### 7.1 The curriculum gate omitted absolute QoS feasibility

`phase2_curriculum_gate_pass` checks episode count, curriculum coverage, finite behavior, collapse, original reward balance, Q differentiation, return convergence, and local stability. It never requires delivery, stale, or fairness thresholds to pass. The notebook therefore accepted and selected infeasible lineages.

### 7.2 “Converged” was interpreted too broadly

The constrained objective stabilized at negative returns. Stabilization was correctly measured, but the gate name allowed it to be read as successful QoS convergence. Convergence and feasibility must be separate gates.

### 7.3 Feasibility evidence and action budget were mismatched

The thresholds were described as feasible because static, energy-proportional, harvest-proportional, and FFSS policies achieved them on identical schedules. Those policies use the environment's 24-slot frame budget, while HTA-MAC was limited to 12 slots. This proved schedule-level feasibility, not budget-12 feasibility.

The budget diagnostic confirms the mismatch: all three targets pass 5/5 at budget 16 but only 1/5 at budget 12.

### 7.4 Training QoS scope did not match evaluation QoS scope

The controller sees episode-cumulative metrics for the current target cluster. Network selection should be based on whole-network delivery, stale ratio, and fairness over the frozen schedule. Dynamic membership also changes which nodes enter the local fairness calculation. This creates a discontinuous, poorly calibrated proxy.

The clearest evidence is fairness: seed 4299 satisfies network fairness on 5/5 schedules, yet its training fairness proxy satisfies `0.95` in only 8% of the final 50 episodes and keeps the fairness multiplier saturated.

### 7.5 Per-step dual updates caused saturation and order dependence

Multipliers persist across shuffled seed/rank episodes, while cumulative counters reset. Early-episode fairness transients and difficult clusters repeatedly increase global multipliers. The next unrelated curriculum case inherits those multipliers. Delivery and fairness multipliers then stay near their cap even when the network metric is already acceptable.

### 7.6 Constraint domination was not part of the domination gate

The reward-balance gate considers only the six original physical terms. In the final 50 episodes, constraint penalties of roughly `-339` to `-371` exceed physical rewards of roughly `269` to `281`. Thus `pathological_reward_domination=false` is true only for the original reward decomposition, not for the actual learning objective.

### 7.7 Selection used training-tail proxies instead of deterministic network evaluation

The chosen score averages positive violations from epsilon-greedy training episodes with changing cluster membership. It does not rank final checkpoints by deterministic network-wide QoS. Seed 4299 remains best after network evaluation, but that agreement is fortunate rather than guaranteed by the selection method.

### 7.8 Local stability did not detect network lifetime regression

The local curriculum evaluator showed stable FND around 143 for episodes 400-500. Network replay showed FND dropping from 268 to 213. Early stopping and checkpoint selection must use the network evaluator, not only local target-cluster summaries.

### 7.9 Provenance was incomplete

The archive's sidecar checksum, checkpoint hashes, and configuration hashes are strong, but the Colab summaries record `git_hash=unavailable` and the results do not contain a source-commit manifest. This does not invalidate the numerical results, but it weakens external reproduction.

## 8. Recovery plan

### Layer R0 — Freeze and protect current evidence

- [x] Verify archive and checkpoint hashes.
- [x] Preserve seed 4299 as the best **budget-12 development candidate**, not as a final model.
- [x] Run network-wide development evaluation for all three final checkpoints.
- [x] Run network snapshot and budget diagnostics.
- [ ] Add the uploaded archive SHA and this report to an immutable development evidence manifest.
- [ ] Keep held-out seeds `3100-3104` locked.

**Exit criterion:** current evidence is reproducible and cannot be silently overwritten.

### Layer R1 — Align metrics and gates

- Add an explicit `qos_feasibility_gate` separate from `convergence_gate`.
- Compute primary QoS from deterministic whole-network development replay at horizon 300.
- Evaluate every stability checkpoint at episodes 400, 450, and 500 network-wide.
- Require the candidate to pass the frozen delivery, stale, and fairness thresholds under the registered budget.
- Report per-seed pass counts in addition to medians.
- Treat target-cluster QoS only as a training surrogate, never as the final selection metric.

**Exit criterion:** a checkpoint cannot receive overall `pass` when any predeclared development QoS gate fails.

### Layer R2 — Establish equal-budget feasibility

- Add equal-budget static, proportional, and random diagnostics at budgets 8, 12, 16, 20, and 24.
- Do not use a 24-slot comparator to assert a 12-slot feasibility claim.
- Freeze budget 12 as the energy/lifetime branch.
- Freeze budget 16 as the proposed QoS-balanced branch, based on the completed development diagnostic.
- Preserve the original thresholds; do not weaken them after seeing these results.

**Exit criterion:** each branch has constraints justified by policies operating under the same action capacity.

### Layer R3 — Repair the constrained controller

- Replace per-step global dual updates with episode-end or full-25-pair epoch updates.
- Aggregate violations across the curriculum before changing shared multipliers.
- Add a fairness warm-up so zero/early cumulative service does not dominate dual updates.
- Normalize penalties per step and calibrate them against the physical reward scale.
- Add anti-windup behavior and log saturation duration.
- Add a constraint-dominance diagnostic that includes the actual Lagrangian penalty.
- Recalibrate the C51 return scale/support for budget 16 before confirmation training.

**Exit criterion:** smoke training is finite, duals do not remain pinned at their cap, and improving network QoS reduces the corresponding dual pressure.

### Layer R4 — Retrain the budget-16 QoS-balanced branch

1. Run a 50-episode seed-2299 smoke with the repaired controller.
2. Audit equivariance, budget/caps, categorical support, collapse, dual saturation, and penalty balance.
3. If the smoke passes, train three registered 500-episode lineages at budget 16.
4. Evaluate episodes 400, 450, and 500 network-wide on seeds `2300-2304`.
5. Select lexicographically:
   - first, joint QoS feasibility under the frozen thresholds;
   - second, robust per-seed pass count;
   - third, FND and energy efficiency;
   - fourth, throughput/fairness tie-breaks.

**Exit criterion:** one development-selected budget-16 checkpoint passes structural gates, absolute QoS gates, network stability, and the predeclared selection rule.

### Layer R5 — Locked held-out evaluation

Only after Layer R4 passes:

- freeze the checkpoint SHA-256, budget, evaluator command, horizon, seeds, and analysis script;
- run held-out seeds `3100-3104` once;
- do not retrain or reselect after viewing held-out results;
- report right-censoring with Kaplan-Meier/common-horizon restricted event-free time;
- report paired non-lifetime differences without converting five development/held-out pairs into broad superiority claims.

**Exit criterion:** a final censor-aware held-out report with no post-selection leakage.

## 9. Recommended immediate decision

Do **not** proceed to held-out Phase 3 with the current budget-12 selection.

Keep seed 4299/B12 as an energy-first reference checkpoint. Implement the metric/gate/controller repairs, calibrate the budget-16 return scale, and retrain a registered budget-16 three-lineage confirmation. Budget 16 is the evidence-backed next branch because it is the smallest tested budget that passes all three QoS targets on all five development schedules while preserving a meaningful efficiency/lifetime advantage over service-heavy baselines.

## 10. Reproduction commands and generated evidence

Final-checkpoint development evaluation template:

```powershell
python -B experiments\run_phase3_pilot.py `
  --seeds 2300,2301,2302,2303,2304 `
  --horizon 300 `
  --run-name <development-run-name> `
  --skip-compatibility `
  --hta-checkpoint <checkpoint-path> `
  --hta-budget 12
```

Generated final-checkpoint evidence:

| Evidence | Summary SHA-256 |
|---|---|
| Seed 2299/B12 | `B71D134BAFBDF56065A47728171F79979B95F558576CFCFE8A9F6B14DA566AAA` |
| Seed 3299/B12 | `AAAD603625A7791454683627EFC01400D20F612BDDA38CFF93A99157B8A4A3DA` |
| Seed 4299/B12 | `3CC5665BD00A56A91525E7B11C3087B488C755A18FE156D1F82D358BA1670227` |
| Seed 4299 episode 400/B12 | `875386BAA4A52061B17F4406989771EAF8C4B2ADB215697FAA862C41B9EF7C5F` |
| Seed 4299 episode 450/B12 | `A093C391519FF04ECADA3AA00B5BD9B4C2F3390D3A25847561F9B4A815101812` |
| Seed 4299/B16 diagnostic | `9FAB178775BE05EA4F6004815A6D7384AA70EBE81088B1E7F19F49BCD2FE3D48` |
| Seed 4299/B20 diagnostic | `0BE19FE4234F39DF33A24563EE433B62ABAEF392FBE3F439F521DABA8E80CD23` |
| Seed 4299/B24 diagnostic | `88CBCF854BB5A5C910712CF452788DEBE6523EBBA8D5ABCB8758212E45` |

All generated outputs are under `outputs/phase3/` and are development-only.

## 11. Safe claims and prohibited claims

Safe now:

- all three lineages trained and passed structural audits;
- seed 4299 is the best current budget-12 development candidate;
- the budget-12 candidate improves lifetime/throughput/energy-efficiency versus several references on development schedules;
- budget 16 is the smallest tested diagnostic budget satisfying all frozen QoS thresholds on 5/5 development schedules;
- a lifetime/energy versus delivery/freshness trade-off remains.

Not safe now:

- “Phase 2D QoS passed”;
- “the model satisfies all QoS constraints at budget 12”;
- “convergence proves feasibility”;
- “seed 4299 is the final model”;
- any held-out, statistically significant, or publication-level superiority claim.
