# HTA-MAC Paper-Aligned B16 QoS-Repaired Results Audit

**Audit date:** 8 August 2026  
**Audited folder:** `HTA_MAC_PaperAligned_B16_QoSRepaired_Trained_Results_20260808/`  
**Audited archive:** `HTA_MAC_PaperAligned_B16_QoSRepaired_Trained_Results_20260808.zip`  
**Track:** secondary paper-aligned B16 development side study  
**Claim boundary:** paper-aligned comparison, not a third-party reproduction, not confirmation, and not a replacement for the registered idle-on hybrid solar/thermal track

## 1. Executive verdict

The Colab run completed successfully and the QoS accounting repair worked. All three 500-episode training lineages passed the existing Phase 2 structural/learning gate, all three permutation-foundation audits passed, all 105 short-horizon policy trials completed, optimizer seed 7399 was selected by the frozen development rule, the action-distinctness audit passed, and all 35 selected-policy long-horizon trials completed.

The most important distinction is:

- **Accounting success:** confirmed. Across all 1,500 training episodes, delivered target packets never exceeded target packets offered. This closes the pre-repair cohort-consistency defect.
- **Model-performance improvement:** not demonstrated. At 300 rounds the selected repaired checkpoint is essentially tied with the energy-proportional comparator and reproduces the pre-repair checkpoint's near-ceiling result. At 3,000 rounds it has strong delivery relative to conservative baselines, but its first-node-death time is earlier than every comparator and its energy efficiency is below the aggressive proportional/random comparators.
- **Statistical superiority:** not supported. No reported paired comparison reaches two-sided `p < 0.05`; with five nonzero paired differences, the exact Wilcoxon floor is `p = 0.0625`.
- **Publication status:** this is useful development evidence and a valid repair artifact, but it is not yet evidence for a global superiority, lifetime, constrained-RL, C1, or C3 publication claim.

The correct decision is to retain seed 7399 as the frozen **development candidate**, run an untouched confirmation design, and repair the training gate/objective mismatch before claiming that constrained training itself is effective.

## 2. Integrity, provenance, and scope

### 2.1 Archive checks

- ZIP SHA-256: `7ba23f606d1943e97221c43d13564bd04618359f6aee3a876c54878269366ae2`
- ZIP file entries: 39
- ZIP CRC check: passed
- Extracted-folder comparison: 39/39 entries present and byte-identical to the ZIP
- Result files found in the extracted folder: 39

The included `COLAB_PAPER_ALIGNED_B16_QOS_REPAIRED_MANIFEST.json` is the **training-input bundle manifest**, not a generated-results manifest. It freezes 184 source/input files, but it does not enumerate the 39 output files. The ZIP hash above is therefore the present end-to-end integrity anchor for this particular result transfer.

### 2.2 Frozen experiment identity

| Item | Value |
|---|---:|
| Optimizer seeds | 5399, 6399, 7399 |
| Development environment seeds | 2400–2404 |
| Reserved confirmation seeds | 3400–3404 |
| Prohibited registered held-out seeds | 3100–3104 |
| Training episodes per lineage | 500 |
| Training horizon | 300 rounds |
| Long evaluation horizon | 3,000 rounds |
| Architecture | `equivariant_set_branching` |
| Observation schema | `phase2d_ttl_cap_v2`, 58 features |
| Branch actions | 4 |
| Frame-slot budget | 16 per active cluster-round |
| Trainable parameters | 115,123 |
| Environment profile SHA-256 | `57e579e702dfcdab0d0df78cb20c0944212b7d34d6d32667df352e3cbe88f964` |
| QoS configuration SHA-256 | `a1c70120d3b31c36cab15e969e99e66f041b1b8609828b0aa0f7524fc1e63577` |
| Reward scale | 0.08354662726692148 |

The seed firewall held. The only episode/evaluation seeds in the fresh Phase 2 and Phase 3 artifacts are 2400–2404. Neither 3100–3104 nor 3400–3404 appears as an executed seed. This also means the evaluation is development-set evaluation, not independent confirmation.

The profile deliberately disables idle-listening energy and thermal harvesting. It therefore does not evaluate the registered primary-track C3 idle-listening contribution or C1 hybrid-harvesting contribution. Any statement that “the death penalty is inert” is B16-profile-specific: idle listening, previously a major depletion mechanism, is disabled here.

### 2.3 Provenance limitation

The manifest records parent Stage 2 commit `e26011e67e95b4d712fbbf15704bdb5bd78f67de`, but records `hta-mac` as untracked (`?? hta-mac/`), and the generated summaries report `git_hash: unavailable`. The input-file manifest provides content-level source evidence, but conventional commit-level provenance is incomplete. A publication artifact should include a committed HTA-MAC source tree and a generated-output manifest.

The Phase 3 field `frozen_checkpoint_sha256 = ccb572...` refers to the frozen schedule/source checkpoint from the manifest. The trained HTA-MAC checkpoint is separately and correctly identified by `trained_checkpoint_sha256`; these fields should be renamed in a future schema to avoid ambiguity.

## 3. Did the QoS repair work?

Yes, as an accounting repair.

The frozen contract is now:

- ratio scope: `episode_cumulative_target_backlog_service`
- demand field: `target_packets_offered`
- fairness metric: `target_cluster_service_fairness`
- training thresholds: delivery at least 0.55, stale ratio at most 0.45, fairness at least 0.70

Independent parsing of all episode logs gave:

| Optimizer seed | Episodes satisfying delivered <= offered | Largest delivered-minus-offered value | Joint QoS passes, all 500 | Joint passes, last 50 | Median delivery, all / last 50 | Median fairness, all / last 50 |
|---:|---:|---:|---:|---:|---:|---:|
| 5399 | 500/500 | -913 | 264/500 | 0/50 | 0.5657 / 0.5023 | 0.9413 / 0.9451 |
| 6399 | 500/500 | -960 | 316/500 | 1/50 | 0.5747 / 0.5147 | 0.9378 / 0.9310 |
| 7399 | 500/500 | -826 | 460/500 | 47/50 | 0.6396 / 0.6235 | 0.9454 / 0.9363 |

All 1,500 episodes satisfy the physical inequality. The prior impossible pattern—more target deliveries than target demand—does not recur.

However, the repaired accounting exposes a second issue: **the existing Phase 2 gate does not require QoS satisfaction.** It requires full curriculum coverage, finite values, non-collapse, non-dominating reward, Q differentiation, reward convergence, and policy stability. Consequently, seeds 5399 and 6399 pass Phase 2 even though their final 50 episodes almost never jointly satisfy the training constraints. This is not a failure of the repaired metric; it is a gate-definition gap.

Additional training findings:

- Stale constraints were easy in all lineages: all 1,500 episodes passed the 0.45 stale threshold.
- Fairness generally passed, while delivery was the active failure mode for 5399 and 6399.
- Seed 7399 ended with all Lagrange multipliers at zero and a last-50 joint pass rate of 94%.
- Seed 5399 ended with delivery multiplier 0.419; seed 6399 ended at 0.293. These nonzero values are consistent with unresolved delivery violations.
- The death reward term fired zero times in all 1,500 training episodes (450,000 environment steps). This is expected under the 300-round, idle-off B16 profile and must not be generalized to the primary track.
- The packet-delivery reward was dominant but below the pathological threshold: approximately 72% of absolute last-50 reward contribution in seed 5399, with the summaries marking pathological domination false.

## 4. Architecture and learning-foundation checks

All three checkpoints passed the foundation audit:

- 20 random permutations and 10 targeted swaps per lineage
- maximum Q error: 0
- maximum log-probability error: 0
- projected-allocation agreement: 1.0
- local-argmax agreement: 1.0
- all allocations feasible
- fraction of Q values within one atom of a C51 boundary: 0
- median boundary mass: 0.0295–0.0299, below the 0.10 gate

The trajectory Q checks also differentiated the same node/head under low- and high-harvest trajectories:

| Seed | Maximum absolute Q difference | Differentiated |
|---:|---:|---:|
| 5399 | 0.05450 | yes |
| 6399 | 0.04414 | yes |
| 7399 | 0.05042 | yes |

All lineages completed 150,000 environment steps, contained no non-finite values, avoided all-sleep collapse, passed reward convergence, and passed the three-snapshot policy-stability gate.

These results support architectural correctness and numerical stability. They do not by themselves establish policy superiority.

## 5. Development selection

All three checkpoints passed the frozen 300-round global gates on all five development seeds:

- delivery ratio >= 0.95
- stale-drop ratio <= 0.01
- global service fairness >= 0.95

The predeclared ranking selected seed 7399:

| Rank | Optimizer seed | Joint global passes | Median violation | Median throughput | Median packets/J |
|---:|---:|---:|---:|---:|---:|
| 1 | **7399** | 5/5 | 0 | **29,980** | 2,100.41 |
| 2 | 5399 | 5/5 | 0 | 29,979 | 2,100.11 |
| 3 | 6399 | 5/5 | 0 | 29,978 | 2,100.38 |

This selection is procedurally correct but practically fragile: only one to two median packets separate the three lineages. Seed 7399 should be frozen because the rule selected it, not because the data show a meaningful optimizer-seed advantage.

Selected checkpoint SHA-256: `829a24ac9f43889b7ce0dae54364db8aaaeab2cfd89e4768cfc79785e6854f5c`

## 6. Selected policy at 300 rounds

Values below are medians over development seeds 2400–2404.

| Policy | Throughput | Delivery | Stale ratio | Service fairness | Packets/J | Budget use | Budget binding | Demand contention |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **HTA-MAC 7399** | **29,980** | **0.9960** | **0** | 0.999998 | 2,100.41 | 0.3123 | 0.0178 | 0.0128 |
| Energy proportional | 29,979 | 0.9960 | 0 | 0.999998 | 2,100.65 | 0.6174 | 0.2560 | 0.0143 |
| Harvest proportional | 29,912 | 0.9938 | 0.00179 | 0.999984 | 2,101.77 | 0.6174 | 0.2560 | 0.0208 |
| Random budgeted diagnostic | 29,945 | 0.9949 | 0.00080 | 0.999994 | 2,101.71 | 0.6174 | 0.2560 | 0.0210 |
| Static equal | 23,967 | 0.7962 | 0.1905 | 0.999994 | 2,047.29 | 0.2497 | 0.0033 | 0.2512 |
| FFSS adapted | 23,967 | 0.7962 | 0.1905 | 0.999989 | 2,047.19 | 0.2497 | 0.0033 | 0.2512 |
| S2A2MAC adapted | 23,130 | 0.7684 | 0.2219 | 0.993772 | **2,180.54** | 0.2542 | 0.0603 | 0.1777 |

Interpretation:

1. HTA-MAC clearly serves more traffic than static-equal, adapted FFSS, and adapted S2A2MAC over 300 rounds.
2. It is effectively tied with energy-proportional: the median paired throughput advantage is one packet and delivery advantage is 0.000033. Neither is significant (`p = 0.25`). HTA-MAC is slightly less energy-efficient on every seed (median difference -0.293 packets/J, `p = 0.0625`).
3. Compared with harvest-proportional and the random floor, HTA-MAC has small, directionally consistent delivery gains, but slightly lower energy efficiency. The best attainable reported `p` remains 0.0625.
4. HTA-MAC uses roughly half the nominal slot budget of the aggressive proportional/random policies while achieving similar delivery. This demonstrates less over-allocation, not a corresponding energy win: with idle energy disabled and radio cost largely packet-driven, its median consumed energy remains almost identical.
5. Low contention for successful policies means the B16 short horizon is not strongly budget-limited. Under-serving static/FFSS/S2A2MAC policies accumulate queues, which raises their later demand-contention fractions. Therefore the near-ceiling result mostly shows adequate service in an easy short-horizon regime, not superiority under sustained budget pressure.
6. Static-equal and adapted FFSS are behaviorally and metrically almost indistinguishable here. The runner itself states that the environment cannot represent FFSS slot ordering; this is an adaptation, not a reproduction of full FFSS.

### 6.1 Comparison with the pre-repair checkpoint

The copied preflight evaluates the old seed-5299 pre-repair checkpoint on the same five development seeds. Its medians versus the new seed-7399 checkpoint are:

| Metric | Pre-repair 5299 | Repaired 7399 | Change |
|---|---:|---:|---:|
| Throughput | 29,980 | 29,980 | 0 |
| Delivery ratio | 0.996013 | 0.996013 | 0 |
| Stale ratio | 0 | 0 | 0 |
| Service fairness | 0.999998 | 0.999998 | 0 |
| Energy consumed | 14.27306 J | 14.27342 J | +0.00036 J |
| Energy efficiency | 2,100.46 | 2,100.41 | -0.053 packets/J |

The repair materially improves metric validity, but it does not improve these external 300-round outcomes. This is still valuable: trustworthy unchanged performance is stronger evidence than an invalid apparent gain. It must not be described as a QoS performance improvement.

## 7. Action-distinctness audit

The selected policy is not action-identical to any comparator over 1,500 common-state observations:

| Comparator | Exact round-action agreement | Mean normalized L1 | Active-set Jaccard |
|---|---:|---:|---:|
| Static equal | 0.0033 | 0.2024 | 0.9971 |
| FFSS adapted | 0.0033 | 0.2023 | 0.9973 |
| Energy proportional | 0 | 0.5137 | 0.9973 |
| Harvest proportional | 0 | 0.5749 | 0.9390 |
| Random budgeted diagnostic | 0 | 0.5787 | 0.9362 |
| S2A2MAC adapted | 0 | 0.8690 | 0.4692 |

This closes the policy-identity concern: HTA-MAC chooses different slot counts even when it activates nearly the same node set as energy-proportional. It does **not** establish better outcomes; the 300-round outcome remains a practical tie.

The audit reports approximately 99.93 total selected slots per round across about 20 active clusters, not 99.93 slots in one cluster. The per-cluster budget remains 16.

## 8. Selected policy at 3,000 rounds

All FND and HND events were observed for all policies and all five seeds. Thus FND/HND are not censored at this horizon, although complete network death remains right-censored because nodes are still alive at round 3,000.

| Policy | Throughput | Delivery | Stale ratio | Packets/J | FND | HND | Alive at 3,000 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **HTA-MAC 7399** | 140,742 | **0.8294** | **0.1694** | 2,086.06 | **1,080** | 1,392 | 19 |
| Energy proportional | 140,336 | 0.8475 | 0.1511 | 2,091.31 | 1,090 | 1,389 | 17 |
| Harvest proportional | 140,469 | 0.8403 | 0.1584 | 2,092.03 | 1,113 | 1,383 | 18 |
| Random budgeted diagnostic | 140,432 | 0.8435 | 0.1552 | 2,093.34 | 1,088 | 1,374 | 17 |
| Static equal | 146,801 | 0.6599 | 0.3384 | 2,009.45 | 1,376 | 1,996 | 32 |
| FFSS adapted | 146,416 | 0.6609 | 0.3373 | 2,009.55 | 1,385 | 1,983 | 30 |
| S2A2MAC adapted | **162,885** | 0.6877 | 0.3110 | **2,166.73** | **1,594** | **2,227** | **32** |

### 8.1 What this means

- HTA-MAC maintains far better delivery and lower stale-drop ratios than static-equal, adapted FFSS, and adapted S2A2MAC.
- It does not dominate energy-proportional, harvest-proportional, or the random diagnostic. Against energy-proportional, HTA-MAC's median delivery is lower by 0.0182 and efficiency lower by 6.51 packets/J; both directions occur on all five seeds (`p = 0.0625`).
- HTA-MAC has the earliest median FND of every policy. Energy-proportional is +10 rounds, random +8, harvest +33, static +296, FFSS +305, and S2A2MAC +514 by the displayed medians.
- HTA-MAC HND is approximately tied with aggressive policies, but far earlier than conservative policies: static and FFSS are about 600 rounds later and S2A2MAC about 835 rounds later in paired comparisons.
- Raw throughput is not interchangeable with delivery ratio in this long run. Policies that keep more nodes alive generate more traffic, so static/FFSS/S2A2MAC can deliver more total packets while dropping a much larger fraction. The evaluation reveals a genuine service-versus-lifetime trade-off, not one universally best policy.
- The death penalty was inert during 300-round training, yet FND occurs near round 1,080 during evaluation. The policy therefore received no direct training experience about the failure event used for the long-horizon lifetime claim. This is the main objective/horizon mismatch.

No long-horizon paired result supports inferential superiority. Directionally uniform comparisons generally produce `p = 0.0625`, not less than 0.05.

## 9. What succeeded

1. The target demand/delivery cohort bug is fixed across every logged episode.
2. The equivariant architecture passes current-checkpoint permutation and projection audits exactly.
3. Training is numerically stable and reproducible across three optimizer lineages.
4. The seed firewall is intact.
5. Development selection follows the predeclared rule rather than post-hoc preference.
6. The chosen policy is action-distinct from all comparators.
7. The long horizon resolves FND and HND instead of incorrectly treating 300-round censoring as lifetime.
8. Allocation-pressure telemetry explains why near-ceiling short-horizon outcomes occur.
9. Scope and baseline-adaptation limitations are recorded in the machine-readable artifacts.

## 10. What did not succeed, and why

### 10.1 No demonstrated gain from the QoS retraining

The selected checkpoint duplicates the old checkpoint's 300-round external metrics to practical precision. The repair fixes what the optimizer measures, but the evaluation regime is easy enough that several aggressive policies already achieve near-ceiling service.

### 10.2 The training gate ignores constraint attainment

Two lineages pass Phase 2 despite 0/50 and 1/50 joint QoS passes in their final windows. Reward convergence is not constraint convergence. The current gate can certify a stable policy while the configured constraint remains violated.

### 10.3 The short training horizon cannot teach lifetime behavior

No training death occurs in 450,000 logged steps. Long-evaluation FND occurs around 1,080 rounds. A death penalty that never fires cannot shape pre-death energy-risk behavior.

### 10.4 Development evaluation is not independent

Seeds 2400–2404 generate the curriculum and are reused for model selection/evaluation. This is allowed by the development protocol, but it cannot estimate confirmation generalization.

### 10.5 The B16 environment is weakly budget-constrained at 300 rounds

Successful policies see only roughly 1–2% demand-contention rounds. This compresses outcome differences and makes it difficult to demonstrate the benefit of an adaptive scheduler.

### 10.6 Comparator and claim limitations remain

FFSS and S2A2MAC are environment-compatible adaptations. Missing published artifacts and unrepresentable mechanisms prevent literal reproduction. The B16 profile also disables the primary idle-listening and thermal-harvesting contributions.

### 10.7 Provenance is content-frozen but not commit-clean

The exact training inputs are hashed, but HTA-MAC is untracked relative to the recorded parent commit and generated outputs lack their own manifest.

## 11. Required next sequence

### Step 1 — Freeze this development result

- Preserve seed 7399 and checkpoint SHA-256 `829a24...f5c` as the selected development candidate.
- Do not replace it with a post-hoc “better-looking” seed.
- Preserve the ZIP with SHA-256 `7ba23f...6ae2`.
- Label every result as development-only and B16-specific.

### Step 2 — Repair the Phase 2 acceptance gate before another training study

For a newly preregistered experiment version, add constraints such as:

- all 1,500 episode records must satisfy `delivered <= offered`;
- final-window joint QoS pass fraction must meet a frozen minimum, for example at least 90%;
- final-window median delivery, stale ratio, and fairness must each meet their thresholds;
- report final multipliers and reject unresolved upward pressure rather than accepting reward convergence alone;
- preserve architecture, non-collapse, reward-balance, Q-differentiation, and stability gates.

This change must define a new experiment version; it must not retroactively alter the acceptance status of the completed run.

### Step 3 — Align training with the lifetime endpoint

Before claiming lifetime benefit, create a development profile in which the agent experiences pre-death/death risk during training. Options to preregister and ablate include:

- longer or variable training horizons extending beyond typical FND;
- an energy-risk surrogate based on lower-tail normalized residual energy or predicted time-to-depletion;
- a penalty for worsening minimum/p10 residual energy, not only deaths after they occur;
- temporal curriculum from 300 rounds to an FND-covering horizon;
- explicit service-lifetime multi-objective constraints so lower delivery cannot masquerade as lifetime improvement.

The trajectory-order and concavity losses should remain. Rank/percentile energy features can be evaluated as a separate preregistered ablation after the endpoint mismatch is fixed.

### Step 4 — Use a genuinely discriminative development stress test

Build a separate, clearly labeled stress profile with materially higher budget contention through frozen traffic/queue/budget settings. Do not silently change B16 and continue calling it the same experiment. Require that the stress profile preserve physical packet accounting and compare all policies on common exogenous schedules.

### Step 5 — Run untouched confirmation

- Do not use prohibited seeds 3100–3104.
- The reserved 3400–3404 seeds may provide a five-seed confirmation check for the already-frozen candidate, but five seeds cannot yield an exact two-sided Wilcoxon value below 0.05 when all nonzero differences share one direction.
- For inferential claims, preregister a larger independent confirmation seed set, the primary endpoint, alpha, multiplicity correction, effect-size threshold, and failure rule before observing it.
- Evaluate both 300-round service and 3,000-round FND/HND trade-offs. Never select a model on the confirmation results.

### Step 6 — Return to the registered primary track

Run the frozen candidate or a separately preregistered successor in the idle-on hybrid solar/thermal environment. Only that track can test C1 and C3. Re-measure death-penalty firings there; do not carry the B16 zero-death conclusion into the primary-track reward narrative.

### Step 7 — Publication packaging

- Commit the exact HTA-MAC source tree.
- Generate a results manifest containing relative path, byte count, and SHA-256 for every output.
- Record runtime/library/GPU metadata and the exact command line.
- Include raw paired data, censor-aware analyses, baseline adaptation disclosures, and null/failed gates.
- Present Pareto trade-offs among delivery, stale drops, FND/HND, throughput, and energy efficiency; do not collapse them into an unsupported single “better” claim.

## 12. Claim matrix after this run

| Candidate claim | Status | Reason |
|---|---|---|
| QoS cohort accounting is repaired | **Supported** | 1,500/1,500 episodes satisfy delivered <= offered |
| Equivariant architecture is implementation-correct | **Supported within audited states** | exact permutation/projection audit pass for all checkpoints |
| Seed 7399 is the frozen development candidate | **Supported** | selected by predeclared ranking |
| HTA-MAC is action-distinct from comparators | **Supported on development trajectories** | no comparator has identical signatures |
| HTA-MAC meets 300-round global QoS gates | **Supported on development seeds** | 5/5 passes |
| QoS repair improved 300-round performance | **Not supported** | metrics are effectively unchanged from pre-repair checkpoint |
| HTA-MAC beats energy-proportional | **Not supported** | practical short-horizon tie; long-horizon trade-off; no `p < 0.05` |
| HTA-MAC improves lifetime | **Not supported** | earliest median FND; HND only tied with aggressive policies |
| Constrained RL is robust across optimizer seeds | **Not supported** | two lineages fail final-window target QoS despite Phase 2 gate pass |
| C1 hybrid-harvesting contribution | **Not evaluated** | thermal harvesting disabled |
| C3 idle-listening contribution | **Not evaluated** | idle listening disabled |
| Third-party algorithm reproduction | **Not supported** | paper-aligned adaptations with disclosed representational limits |
| Publication-ready global superiority | **Not supported yet** | development seeds, n=5, no confirmation, unresolved trade-offs |

## 13. Bottom line

This run successfully turns an invalid constrained-training signal into a physically consistent one and produces a defensible frozen development candidate. That is a meaningful research milestone.

It also reveals that the current result is primarily a **validity repair**, not a performance advance. Seed 7399 provides excellent short-horizon service with economical slot allocation, but it does not beat the strongest simple comparators, and its long-horizon FND is worse. The next experiment should therefore target the objective/horizon and contention-regime mismatches, then evaluate the already-frozen candidate on untouched confirmation seeds. Any paper should report the service-versus-lifetime Pareto trade-off and the negative findings rather than claiming universal superiority.
