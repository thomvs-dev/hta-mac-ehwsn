# HTA-MAC Paper-Aligned B16 Trained Results Analysis

**Prepared:** 7 August 2026
**Evidence directory:** `HTA_MAC_PaperAligned_B16_Trained_Results_20260806/`
**Environment:** exploratory paper-aligned terrestrial solar EH-WSN, not a third-party reproduction
**Decision scope:** development only; reserved confirmation seeds 3400-3404 remain unused

## 1. Executive verdict

The training run is technically successful but not yet sufficient for a lifetime or superiority claim.

All three 500-episode lineages passed the implemented Phase 2 curriculum gate, convergence gate, policy-stability gate, permutation-equivariance audit, budget/cap feasibility audit, and C51-support audit. The development selector froze optimizer seed 5299, and all five development trials met the transferred delivery, stale-drop, and queue-fairness thresholds.

The strongest whole-network result is high short-horizon service quality:

- median HTA-MAC throughput: 29,980 of 30,100 generated packets;
- median delivery ratio: 0.996013;
- median stale-drop ratio: 0;
- median global queue fairness: 0.999998;
- 100 of 100 nodes alive after 300 rounds in every trial.

However, the experiment does **not** establish a lifetime gain. No FND or HND event occurred for HTA-MAC or any comparator within 300 rounds, so every lifetime observation is right-censored and every restricted mean event-free time is exactly 300 rounds. HTA-MAC is also effectively tied with the energy-proportional baseline and only slightly ahead of the random-budgeted diagnostic at this horizon. Its large gains are confined to the weaker static-equal, FFSS-adapted, and S2A2MAC-adapted service policies.

A training-only QoS accounting issue must be repaired before confirmation: cumulative target delivery can include queued packets generated outside the current target exposure, while the denominator counts only newly generated packets among the current target members. Logged delivered counts can therefore exceed logged generated counts and the controller clamps delivery to 1.0. The Phase 3 global delivery ratio is correctly bounded and remains valid, but the training delivery multiplier is not interpretable as an end-to-end delivery constraint.

**Decision:** retain seed 5299 as the frozen development selection because that was the prespecified output, but do not run reserved confirmation yet. First repair QoS accounting and obtain event-observable long-horizon development evidence.

## 2. Evidence inventory and integrity

The directory contains 32 files totaling 29,900,035 bytes:

- 3 trained final checkpoints;
- 9 stability checkpoints (episodes 400, 450, and 500 for each lineage);
- 3 Phase 2 episode traces with 500 rows each;
- 3 Phase 2 summaries;
- 3 foundation audits;
- 3 Phase 3 raw-trial CSV files with 35 rows each;
- 3 Phase 3 summaries;
- 3 complete training logs;
- 1 frozen reward-scale calibration;
- 1 development-selection record.

All 11 JSON files parsed, all 3 CSV files parsed, and all three logs ended with `PHASE2_CURRICULUM_GATE_PASS=True`. The three final checkpoints are distinct:

| Optimizer seed | Checkpoint SHA-256 |
|---:|---|
| 5299 | `a2c037e234c2b11695b7235b1bb0f893537a199d32e9bec1094d368b37f8613f` |
| 6299 | `d2c048573109d46a6139f6cf4b84f19f2c009f5f4515de1cedee2c54bf4ae39c` |
| 7299 | `78b81939a7bcbf41380ac0709ec09100034bbe7483f3f2db1a33d61d3ccd5da6` |

The logs report `fatal: not a git repository` when recording the runtime commit because the Colab bundle intentionally omitted `.git`. Configuration and checkpoint hashes are preserved, but the trained-results archive alone does not contain a resolved code commit. Future bundles should copy the source-commit value from the bundle manifest into each training summary.

## 3. Frozen experimental configuration

This branch uses:

- 100 static nodes in a 100 m x 100 m field;
- base station at (50 m, 50 m);
- 0.5 J initial energy per node;
- 20 exogenous, balanced, rotating cluster heads per round;
- trained Stage 1 solar HMM;
- thermal harvesting disabled;
- idle-listening energy disabled to match the selected paper assumptions;
- 4,000-bit data packets and 500-bit control packets;
- per-cluster frame budget B=16;
- per-node maximum allocation n_max=3;
- queue capacity 5 packets and TTL 3 rounds;
- equivariant branching C51 with 115,123 trainable online parameters;
- five development environment seeds: 2400-2404;
- three optimizer seeds: 5299, 6299, and 7299;
- 500 episodes and 300 steps per episode, or 150,000 environment steps per lineage.

This environment does not replace the registered frozen HEART-CH experiment. Cluster-head selection is exogenous and HTA-MAC remains the only learned intervention.

## 4. Reward calibration

The development-only calibration used 100 random-budgeted rollouts and 30,000 discounted returns:

| Field | Value |
|---|---:|
| Lower calibration quantile | 5.368047 |
| Upper calibration quantile | 290.758359 |
| Two-sided magnitude q-star | 290.758359 |
| C51 support | [-30, 30], 51 atoms |
| Headroom target | 80% |
| Frozen reward scale | **0.0825427687** |

Only replay/Bellman rewards were scaled. Raw physical reward, delivery, energy, fairness, and lifetime metrics were not scaled. The smaller scale than the previous registered-profile value is consistent with the larger B16 paper-aligned return magnitude; it must not be transferred back to the registered environment.

## 5. Phase 2 training results

| Seed | Gate | Convergence relative change | Greedy throughput | Greedy delivery | Target queue fairness | Mean residual energy (J) | Mean minimum energy (J) |
|---:|:---:|---:|---:|---:|---:|---:|---:|
| 5299 | PASS | 1.310% | 25,024.25 | 0.831370 | 0.708169 | 0.412743 | 0.375683 |
| 6299 | PASS | 1.705% | **25,233.24** | **0.838314** | 0.703138 | 0.411855 | 0.371917 |
| 7299 | PASS | 2.117% | 25,067.73 | 0.832815 | 0.705889 | 0.412558 | 0.374988 |

All lineages:

- completed all 500 episodes and 150,000 steps;
- visited all 100 seed/target-rank curriculum pairs;
- had no non-finite values;
- avoided the always-sleep collapse;
- had no observed node-death episode and no nonzero death reward term;
- used the exact B16 feasibility projection;
- passed the last-three-snapshot stability threshold;
- retained differentiated trajectory-aware Q values.

The final stability-snapshot throughput values were:

- seed 5299: 24,979.37, 25,008.07, 25,024.25;
- seed 6299: 25,217.40, 25,160.94, 25,233.24;
- seed 7299: 25,085.64, 25,033.55, 25,067.73.

Packets delivered remained the largest absolute reward contribution, but stayed below the prespecified pathological-domination threshold: 71.99% for seed 5299, 72.37% for seed 6299, and 72.29% for seed 7299 versus an 80% limit.

The raw/constrained reward decreased from the first 50 to the last 50 episodes in every lineage. This is not itself evidence of divergence because epsilon fell from 1.0 to 0.05, policy service became more selective, stale behavior improved, and the final 100-episode convergence test passed. It does show that “reward increased throughout training” would be an incorrect claim.

## 6. Foundation audits

| Seed | Median boundary-atom mass | Q within one atom of support boundary | Random allocation agreement | Targeted-swap agreement | Feasible allocations |
|---:|---:|---:|---:|---:|:---:|
| 5299 | 0.030044 | 0% | 100% | 100% | Yes |
| 6299 | 0.029556 | 0% | 100% | 100% | Yes |
| 7299 | 0.029651 | 0% | 100% | 100% | Yes |

All permutation log-probability and Q-value errors were exactly zero in the recorded probes. This resolves the earlier C51 support-saturation and branch-identity concerns for this paper-aligned training run. It establishes correct structural behavior; it does not establish superior network performance.

## 7. Whole-network development comparison

The table reports medians across seeds 2400-2404 for the selected seed-5299 checkpoint. Ranges in the raw CSV files are narrow and do not change the ordering.

| Policy | Throughput | Delivery | Stale-drop ratio | Queue fairness | Energy used (J) | Mean residual E (J) | Efficiency (packets/J) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HTA-MAC** | **29,980** | **0.996013** | **0** | **0.999998** | 14.2731 | 0.391634 | 2,100.46 |
| Energy proportional | 29,979 | 0.995980 | 0 | 0.999998 | **14.2708** | 0.391680 | 2,100.65 |
| Harvest proportional | 29,912 | 0.993754 | 0.001794 | 0.999984 | 14.2347 | 0.392079 | 2,101.77 |
| Random budgeted diagnostic | 29,945 | 0.994850 | 0.000797 | 0.999994 | 14.2493 | 0.391938 | 2,101.71 |
| Static equal | 23,967 | 0.796246 | 0.190465 | 0.999994 | 11.6950 | 0.417388 | 2,047.29 |
| FFSS adapted | 23,967 | 0.796246 | 0.190465 | 0.999989 | 11.6956 | 0.417383 | 2,047.19 |
| S2A2MAC adapted | 23,130 | 0.768439 | 0.221927 | 0.993772 | **10.6011** | **0.428132** | **2,180.54** |

### Interpretation

Against static equal, HTA-MAC delivered 6,013 more packets at the median, a 25.09% relative throughput increase and a 19.98 percentage-point delivery increase. It consumed 22.04% more energy, but throughput grew enough to improve packets-per-joule by 2.60%.

Against S2A2MAC adapted, HTA-MAC delivered 29.62% more packets but consumed 34.64% more energy, so its energy efficiency was 3.67% lower. S2A2MAC preserved substantially more residual energy because it served fewer packets. This is a throughput-versus-conservation tradeoff, not uniform dominance.

Against the strongest simple comparator, energy proportional, HTA-MAC gained only one median packet (0.0033%) and consumed about 0.00249 J more at the paired median. Its packets-per-joule was slightly lower. Against the random diagnostic, HTA-MAC gained 34 packets (0.117%) but again used slightly more energy and had slightly lower energy efficiency. Therefore the present evidence supports “near-ceiling delivery” but not “meaningfully better than strong allocation heuristics.”

With only five paired seeds, the two-sided exact Wilcoxon p-value for a consistent nonzero direction bottoms out at 0.0625. No reported comparison reaches p<0.05, no multiplicity adjustment was applied, and the Phase 3 summaries correctly state that no inferential superiority is claimed.

## 8. Lifetime result: entirely censored

Every policy retained all 100 nodes through round 300 on every development seed:

- FND events: 0/5 for each policy;
- HND events: 0/5 for each policy;
- Kaplan-Meier median FND/HND: not reached;
- restricted mean event-free time through round 300: 300 for every policy.

Consequently:

- FND is not “low”; it is unobserved;
- no policy can be ranked by FND or HND;
- the current horizon is too short for a lifetime claim;
- residual-energy differences are descriptive early-horizon tradeoffs, not replacements for observed survival endpoints.

The next evaluation must extend the development horizon until events occur or until a new prespecified censoring horizon is reached.

## 9. Candidate-selection analysis

All three lineages achieved:

- joint QoS pass count 5/5;
- median constraint violation 0;
- median FND field encoded as 0 because FND was censored;
- median throughput 29,980.

Seed 5299 was therefore selected by input order after a complete tie on the frozen ranking fields. The selection is reproducible, but it is not evidence that seed 5299 is intrinsically better than 6299 or 7299. Whole-network medians differ only in small residual-energy quantities. Do not describe 5299 as the “best-performing” seed; describe it as the prespecified development-selected checkpoint after a tied screen.

The `median_t_fnd: 0.0` field in `DEVELOPMENT_SELECTION.json` is a sentinel produced by converting missing FND to zero. It must not be presented as an actual FND of zero rounds.

## 10. QoS accounting finding

The training controller updates on `target_packets_delivered`, `target_packets_generated`, and `target_stale_drops`. In the dynamic wrapper, cluster membership changes with the exogenous CH schedule:

- delivered counts can service backlog accumulated when a node was outside the current target cluster;
- generated counts add one packet only for nodes currently in the target member set;
- the numerator and denominator therefore do not always refer to the same packet cohort.

A concrete seed-5299 episode logged 2,806 delivered packets and 1,192 generated packets, after which the controller clipped delivery ratio to 1.0. All last-50 episode delivery ratios were exactly 1.0 and the delivery multiplier decayed to 0 in every lineage. This means the Phase 2 delivery constraint was effectively inactive and cannot validate a 0.55 end-to-end delivery requirement.

The Phase 3 global ratio uses total network delivered divided by total network generated and is valid. Its approximately 0.996 result independently demonstrates that the trained policies meet the loose 0.55 development threshold. The training-accounting defect nevertheless needs correction before using constrained-RL language in a paper.

Queue fairness also changes scope: the training greedy target-cluster fairness is approximately 0.703-0.708, whereas Phase 3 global cumulative-service fairness is approximately 0.999998. Both can be valid metrics, but they must have distinct names and thresholds rather than being treated as interchangeable.

## 11. Claim boundary

### Supported now

- Three independent B16 training lineages completed and passed the implemented structural gates.
- C51 support occupancy is healthy in the recorded development audits.
- The equivariant model is exactly permutation-consistent in the recorded probes.
- HTA-MAC achieves approximately 99.6% delivery with near-zero stale drops over 300 development rounds.
- HTA-MAC substantially improves service over static-equal, FFSS-adapted, and S2A2MAC-adapted policies in this simulator.

### Not supported now

- Any FND, HND, or lifetime improvement.
- Inferential superiority over any comparator.
- Meaningful advantage over energy-proportional allocation.
- Reproduction of the cited third-party algorithms or their published numerical results.
- Transfer of these results to the registered frozen HEART-CH profile.
- A claim that the training delivery constraint was correctly enforced end to end.

## 12. Required next actions

1. **Repair Phase 2 QoS accounting.** Track packet generation and service for a consistent per-node or per-packet cohort. Separate target-cluster service fairness from global network fairness. Add regression tests that require `delivered <= generated + initial_backlog` for the same cohort and prohibit silent ratio clipping as a substitute for valid accounting.
2. **Recalibrate after the repair.** Because the constraint signal changes the learning reward, regenerate the development-only return scale and retrain fresh lineages. Preserve the current checkpoints as pre-repair evidence.
3. **Run long-horizon development evaluation.** Start with selected seed 5299 and seeds 2400-2404 at a prespecified horizon such as 3,000 rounds. If FND remains unobserved, extend once using a documented rule. Report censor-aware FND/HND and residual energy.
4. **Add a policy-action distinctness audit.** Quantify action agreement and allocation L1 distance between HTA-MAC, energy proportional, harvest proportional, and random budgeted policies. Near-identical throughput currently hides whether HTA learned a distinct policy.
5. **Define a meaningful primary objective.** If delivery is already saturated, choose an energy-aware endpoint such as energy per delivered packet subject to delivery >=0.99 and stale drops <=0.01. Otherwise the B16 environment rewards every aggressive policy similarly.
6. **Repeat development selection using the frozen rule.** Do not post hoc re-rank the current lineages. After repaired retraining, encode an explicit final tie-breaker and treat censored FND as missing/censored rather than zero.
7. **Only then use reserved confirmation seeds 3400-3404.** Confirmation should run once after code, checkpoint, thresholds, horizon, endpoints, and analysis are frozen. Do not touch registered seeds 3100-3104 in this exploratory branch.
8. **Improve provenance.** Copy the bundle source commit and manifest hash into every result summary so Colab runs do not record only `git_hash=unavailable`.

## 13. Stop conditions

Do not advance to confirmation if any of the following remains true:

- delivery accounting mixes packet cohorts or relies on clipping ratios above one;
