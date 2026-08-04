# HTA-MAC Phase 2B Confirmation Audit and C51 Repair Handoff

**Date:** 4 August 2026  
**Repository:** F:\WSN\matlab\stage2\hta-mac  
**Result artifact:** F:\WSN\matlab\stage2\hta-mac\HTA_MAC_Phase2B_Confirmation_Results_20260803  
**Audience:** next research/coding agent, instructor, or artifact reviewer  
**Status:** the hybrid-trajectory mechanism is confirmed on development data, but the model is not publication-ready because seed 2299 is severely C51 upper-support saturated.

---

## 1. Executive handoff

Phase 2B was created because the registered Phase 2 policies produced numerically different S1/S8 Q-values but almost never changed the projected MAC allocation. The repair added physically scaled inputs, a hybrid counterfactual trajectory-order loss, and a diminishing-return loss. Three 125-episode Colab confirmations were then initialized from their corresponding registered budget-12 checkpoints.

All three runs completed without NaNs, crashes, or always-sleep collapse. Across 11,397 paired mid-episode probes, complete hybrid marginal ordering improved from 4.40% to 79.63%. High-harvest counterfactuals received more slots 348 times and fewer slots 25 times, compared with 13 more and 21 fewer before repair. The repaired policies therefore use the hybrid HMM block at the decision layer.

A post-download audit found a separate C51 problem that the confirmation gate did not test. Seed 2299 has median Q = 29.22 on support [-30,30], 63.8% of Q-values exceed 29, and median probability on the top atom is 94.15%. This is severe categorical-support saturation.

**Decision:** do not begin Phase 3 or Phase 4. First calibrate return scale/support, add a boundary-mass gate, run one short seed-2299 smoke test, and repeat the narrow three-seed confirmation only if that gate passes.

---

## 2. Scope that must remain frozen

- Intra-cluster MAC only.
- HEART-CH CH selection remains frozen through exogenous schedule replay.
- No CH retraining, routing, Pointer Network, or HERMES integration.
- Development schedule seeds remain 2300–2304.
- Held-out seeds 3100–3104 must not select support, reward scale, weights, checkpoints, or stopping rules.
- Phase 4 seeds 4000–4029 remain untouched until a revised plan is written and hashed.
- Queue feasibility, TTL expiry, radio accounting, and idle-energy semantics remain unchanged.
- Thermal results validate a simulated dual-source mechanism, not a dataset-trained thermal forecaster.

The old PHASE4_PREREGISTRATION.md is incompatible with Phase 2B because it specifies 500 episodes, random initialization, no warm-starting, and the old unregularized sweep. A Phase 2B/2C addendum is required before held-out evaluation.

---

## 3. Why Phase 2B was necessary

### 3.1 Registered-policy failure

- The original gate accepted any S1/S8 Q difference above a small numerical threshold.
- Local greedy actions changed in none of the 18 registered models.
- Expanded reset probes almost never changed the projected node allocation.
- Reset probes were weak for multi-slot behavior because initial queues commonly capped actions at one packet.

Therefore, different Q-values did not establish trajectory-conditioned allocation.

### 3.2 Input-scale diagnosis

| Input block | Observed scale |
|---|---:|
| Normalized energy | order 1 |
| Hybrid forecast mean | 6.484e-5 to 3.749e-4 |
| Hybrid forecast variance | 1.549e-8 to 4.239e-8 |
| HMM probabilities | approximately 0 to 0.85 |
| Frozen embedding | up to 47.075; standard deviation 7.169 |

The network normalized the flattened global input rather than each physical block. Forecast moments were muted relative to the inherited embedding.

### 3.3 Literature rationale

Structure-aware EH scheduling is established. Sharma, Mastronarde, and Chakareski exploit value structure to accelerate RL for EH sensors ([IEEE TSP 2020](https://doi.org/10.1109/TSP.2020.2973125)). Chen et al. incorporate scheduling structure into action selection and a DRL loss ([IEEE TWC 2024](https://doi.org/10.1109/TWC.2023.3277861)). Constrained monotonic networks support encoding monotonic relationships ([ICML 2023](https://proceedings.mlr.press/v202/runje23a.html)).

The Branching Dueling architecture remains appropriate for a vector of discrete actions with a combinatorial joint space. HTA-MAC adapts Tavakoli, Pardo, and Kormushev ([AAAI 2018](https://doi.org/10.1609/aaai.v32i1.11798); [open manuscript](https://arxiv.org/abs/1711.08946)). It does not invent Branching DQN.

---

## 4. Implemented Phase 2B changes

### 4.1 Physical input scaling

File: agents/branching_dqn.py, method _transform_state_tensor.

\[
\widetilde{\mu}_H=\mu_H/H_{ref},
\qquad
\widetilde{\sigma_H^2}=\sigma_H^2/H_{ref}^2,
\]

where

\[
H_{ref}=3.749392007120307\times10^{-4}\ \mathrm{J}.
\]

The 32-dimensional inherited embedding is normalized independently per node. Energy, queue, previous allocation, cluster size, and transition probabilities retain their bounded semantics. The transform is opt-in and stored in the checkpoint.

### 4.2 Hybrid trajectory-order loss

File: agents/branching_dqn.py, method _trajectory_order_loss.

One active node at equal energy is changed between joint-low and joint-high solar-plus-thermal transition rows and rectified moments. For

\[
\Delta Q_a=Q(a)-Q(a-1),
\]

the loss is

\[
\mathcal{L}_{traj}
=
\operatorname{mean}_a
\max(0,\eta s+\Delta Q_a^{low}-\Delta Q_a^{high}),
\]

with detached marginal scale s and eta = 0.05. This is a soft inductive bias, not a rule that high harvest must always receive more slots.

### 4.3 Diminishing-return loss

File: agents/branching_dqn.py, method _concavity_loss.

\[
\mathcal{L}_{concave}
=
\operatorname{mean}_a
\max(0,\Delta Q_{a+1}-\Delta Q_a).
\]

It encourages diminishing marginal slot values. It does not prove full MDP separability.

### 4.4 Composite objective

\[
\mathcal{L}
=
\mathcal{L}_{C51}
+1.0\mathcal{L}_{traj}
+0.1\mathcal{L}_{concave}.
\]

All components are logged in episodes.jsonl.

### 4.5 Corrected mechanism audit

File: experiments/audit_phase2_mid_episode_hybrid_sensitivity.py.

It uses active nodes with queue cap at least two, holds unrelated state fixed, changes one node between low/high hybrid blocks, evaluates original/repaired models on identical states, and reports local, node, vector, directional, and marginal-order changes. It is a development diagnostic, not held-out performance evidence.

### 4.6 Precision and confirmation

The Colab notebook benchmarked FP32/BF16 on the actual network and selected BF16 only if native, finite, and at least 10% faster. The NVIDIA L4 selected FP32.

The runner froze budget 12, seeds 2299/3299/4299, 125 episodes, learning rate 1e-5, epsilon 0.10→0.03, trajectory weight 1.0, concavity weight 0.1, and snapshots at 50/75/100/125.

The runner omitted a C51 support-occupancy gate.

---

## 5. Training contract

| Parameter | Value |
|---|---:|
| Input dimension | 50 |
| Actions | 4 (0,1,2,3) |
| Budget | 12 |
| Gamma | 0.99 |
| Learning rate | 1e-5 |
| Batch size | 32 |
| Replay capacity | 5000 |
| Warm-up | 256 |
| Target update | 250 |
| C51 support | [-30,30] |
| Atoms | 51 |
| Atom spacing | 1.2 |
| Architecture | shared branching |
| Max branches | 100 |
| Normalization | enabled |
| Trajectory weight | 1.0 |
| Concavity weight | 0.1 |
| Margin fraction | 0.05 |
| Precision | FP32 |
| Episodes | 125 |
| Max steps | 300 |
| Development seeds | 2300–2304 |
| Optimizer seeds | 2299,3299,4299 |

The environment reset reseeds harvest generation. Baseline and repaired evaluations are reproducible for a development seed, but are not held out.

---

## 6. Artifact integrity and provenance

The result root contains 31 files and 692,822,065 bytes. Each run has 125 episode rows, four stability checkpoints, a final checkpoint, summary, decision, hybrid audit, and runtime record. All JSON/JSONL values were finite. All checkpoints loaded and matched the intended configuration.

Registry SHA-256:

b7f918a65f28b7b7ecc0e109fb048c5f28662af1f0bee516657ba2ddc16c74cb

Final checkpoints:

| Seed | SHA-256 |
|---|---|
| 2299 | f67962f4f48871d7a7ba9446f1e528a6ae381305ced4e9207db68551392c8049 |
| 3299 | b1cfbc377be1a5bae6fe0d571cd3af7de06522a0c7da45cb4a8888376632f182 |
| 4299 | 350b443b3ce84f47a2dc6e6377d86c08010c8375091e6610e33019e0a5c6de61 |

Runtime:

| Seed | GPU | Precision | Runtime |
|---|---|---|---:|
| 2299 | NVIDIA L4 | FP32 | 23.47 min |
| 3299 | NVIDIA L4 | FP32 | 24.21 min |
| 4299 | NVIDIA L4 | FP32 | 23.55 min |

Total was approximately 71.24 minutes.

### Provenance gap

Every summary records git_hash = unavailable. The local training bundle SHA-256 is:

85b62a91654b91c525592c6e5fb87927223f8eb57ac53f836e327ab92804e6eb

The downloaded result lacks the bundle manifest, notebook, bundle hash, and source manifest. Phase 2C must place all provenance records inside the result archive.

---

## 7. Stability and loss results

The notebook decisions say confirmation_pass. The generic summaries correctly say smoke_pass and phase2_curriculum_gate_pass = false because the generic gate requires 500 episodes. Never call these original Phase 2 gate passes.

### Convergence

| Seed | Previous 50 | Last 50 | Relative change |
|---|---:|---:|---:|
| 2299 | 209.240 | 211.410 | 1.04% |
| 3299 | 238.211 | 236.858 | 0.57% |
| 4299 | 236.009 | 231.824 | 1.77% |

### Snapshot spans

| Seed | FND | Throughput | Fairness |
|---|---:|---:|---:|
| 2299 | 1.35% | 3.02% | 3.14% |
| 3299 | 0.34% | 0.67% | 0.40% |
| 4299 | 0.11% | 0.59% | 2.45% |

Packet delivery contributes about 60% of absolute reward; no term exceeds 80%. Death contributes below 0.5% of normalized absolute reward, but the event is not absent or rare in the training curriculum. A reproducible audit of `episodes.jsonl` found 83, 87, and 91 death events for seeds 2299, 3299, and 4299, respectively. The effective Phase 2B death weight recorded by every run is 2.0—not the original design value 10 assumed in the feedback—so any later change back to 10 would be a reward redesign requiring a new calibration and confirmation. Death occurred in 78/125 (62.4%), 84/125 (67.2%), and 87/125 (69.6%) episodes; median deaths per episode were 1, 1, and 1. Thus `w3` is active. Its small normalized share is a relative-magnitude/accounting fact, not evidence that the 300-step horizon makes death unreachable.

Structural losses fell sharply:

| Seed | Trajectory initial→final | Concavity initial→final |
|---|---:|---:|
| 2299 | 0.0313→0.000372 | 0.537→0.000341 |
| 3299 | 0.0443→0.000094 | 0.688→0.000078 |
| 4299 | 0.0330→0.000025 | 0.497→0.000037 |

The imposed structure was learned, but that does not certify the C51 distribution.

---

## 8. Development performance

| Seed | Target packets base→new | FND base→new | Throughput base→new | Fairness base→new | Delivery base→new |
|---|---:|---:|---:|---:|---:|
| 2299 | 1223.44→1425.16 | 141.28→138.52 | 10409.48→10336.96 | 0.4313→0.5563 | 0.6730→0.6894 |
| 3299 | 1286.88→1529.44 | 144.68→142.44 | 10553.00→10749.04 | 0.4482→0.6644 | 0.6769→0.6931 |
| 4299 | 1226.08→1499.68 | 141.32→144.08 | 10369.60→10578.00 | 0.4383→0.6226 | 0.6739→0.6922 |

| Metric | Median change | Range |
|---|---:|---:|
| Target packets | +18.85% | +16.49% to +22.32% |
| FND-free | -1.55% | -1.95% to +1.95% |
| Throughput | +1.86% | -0.70% to +2.01% |
| Fairness | +42.05% | +28.97% to +48.26% |
| Delivery | +2.44% | +2.39% to +2.73% |
| Zero-action fraction | -46.65% | -51.44% to -40.42% |

Service, fairness, and delivery improve. Lifetime is mixed. Reduced sleeping may explain the QoS/lifetime trade-off. No lifetime-dominance claim is allowed.

---

## 9. Mechanistic hybrid results

Across 11,397 backlog-eligible probes:

| Diagnostic | Registered | Repaired |
|---|---:|---:|
| Local argmax changed | 43 (0.38%) | 575 (5.05%) |
| Probed-node slots changed | 34 (0.30%) | 373 (3.27%) |
| Joint vector changed | 280 (2.46%) | 2314 (20.30%) |
| High trajectory got more | 13 (0.11%) | 348 (3.05%) |
| High trajectory got fewer | 21 (0.18%) | 25 (0.22%) |
| All marginals ordered | 502 (4.40%) | 9075 (79.63%) |

Per seed:

| Seed | Registered ordering | Repaired ordering |
|---|---:|---:|
| 2299 | 5.00% | 65.21% |
| 3299 | 2.98% | 86.08% |
| 4299 | 5.23% | 87.71% |

This is the strongest Phase 2B result. It establishes development-state mechanism response, not universal monotonic optimality, held-out performance, or cross-paper superiority.

---

## 10. Critical C51 support failure

C51 uses a fixed categorical support and projects Bellman-updated distributions onto it. Values outside the endpoints are clipped before projection. See Bellemare, Dabney, and Munos ([ICML 2017](https://proceedings.mlr.press/v70/bellemare17a.html)) and Rowland et al. ([AISTATS 2018](https://proceedings.mlr.press/v84/rowland18a.html)). Rainbow does not automatically justify HTA-MAC endpoints because reward preprocessing/task scale differ ([Hessel et al.](https://arxiv.org/abs/1710.02298)).

Current code uses 51 atoms on [-30,30] and clamps targets there, with no reward scaling.

### Repaired audit

| Seed | Median Q | Q max | Q>28 | Q>29 | Median top atom | Top max |
|---|---:|---:|---:|---:|---:|---:|
| 2299 | 29.222 | 29.999 | 78.89% | 63.79% | 94.15% | 99.99% |
| 3299 | 16.384 | 20.996 | 0% | 0% | 11.97% | 29.01% |
| 4299 | 14.892 | 19.308 | 0% | 0% | 7.37% | 20.73% |

### Registered audit

| Seed | Median Q | Q max | Q>28 | Median top atom |
|---|---:|---:|---:|---:|
| 2299 | 20.277 | 29.960 | 26.05% | 46.67% |
| 3299 | 17.978 | 23.916 | 0% | 18.83% |
| 4299 | 16.409 | 20.971 | 0% | 12.09% |

Seed 2299 was partially saturated and worsened. The Bellman target loses resolution, Q differences compress, and auxiliary losses may drive remaining action separation. Seed 2299 is not a valid final C51 checkpoint.

---

## 11. Proposed Phase 2C changes

These changes are proposed, not implemented.

### 11.1 Empirical discounted-return audit

Create experiments/audit_phase2c_return_support.py and compute:

\[
G_t=\sum_{k=0}^{T-t-1}\gamma^k r_{t+k}.
\]

Archive raw rewards/returns and report min/max, median/IQR, p0.5/p1/p95/p99/p99.5, fraction outside support, boundary-atom mass, and fraction of Q within one atom of a boundary. Use development seeds only.

### 11.2 Preferred repair: positive reward scaling

Keep 51 atoms on [-30,30], but scale only replay/C51 reward:

\[
r'_t=c r_t,
\qquad
c=\min\left(1,\frac{0.8V_{max}}{Q^*}\right),
\qquad
Q^*=\max_{j\in\{2300,\ldots,2304\}} Q_{0.995}\!\left(G\mid j\right).
\]

The 0.8 headroom is a proposed engineering rule, not a published constant. `Q*` is the fattest-tail development-seed quantile, not an average and not a per-training-lineage value. Compute it once from the complete fixed reference-rollout set for development schedule seeds 2300–2304, freeze one `c`, and use that identical constant for every Phase 2C smoke, ablation, three-seed confirmation, and later budget arm. A per-seed or per-checkpoint scale is forbidden because it would make the three-seed confirmation compare different effective tasks.

Requirements:

- log raw and scaled reward;
- scale only replay/C51 reward;
- never scale physical metrics;
- store reward_scale in configuration and checkpoint;
- store the five per-seed quantiles, Q*, sample counts, reference-policy/checkpoint hashes, and the pooled return CSV hash beside the locked constant;
- reject checkpoint/scale mismatch.

A positive uniform multiplier preserves exact expected-return action ordering while fitting numerical returns into support.

If boundary-mass creep returns during longer training, do not repeatedly hand-tune c. PopArt is the principled fallback for adaptively normalizing changing value-target scales while preserving unnormalized outputs (van Hasselt et al., NeurIPS 2016). It is not adopted in the first Phase 2C repair because it would add algorithmic scope; it becomes the predeclared escalation only if the single frozen scale fails its support gate.

### 11.3 Alternative: wider support

If scaling is rejected:

\[
V_{max}\ge1.2Q_{0.995}(G),
\]

and preserve approximately 1.2 atom spacing:

\[
N_{atoms}=1+\left\lceil\frac{V_{max}-V_{min}}{1.2}\right\rceil.
\]

Do not widen to [-30,150] with only 51 atoms; spacing would become 3.6.

### 11.4 Reinitialize categorical outputs

Do not reinterpret old logits under a new scale/support.

Preserve normalized input and shared trunk if desired. Reinitialize final value and advantage output layers, corresponding target layers, and their optimizer state. Synchronize target from online and start empty replay.

### 11.5 Scale-invariant structural losses

Normalize auxiliary penalties by detached marginal magnitude:

\[
\widetilde{\mathcal{L}}_{traj}
=
\frac{\operatorname{mean}\max(0,\eta s+\Delta Q^{low}-\Delta Q^{high})}
{\operatorname{stopgrad}(s)+\epsilon},
\]

\[
\widetilde{\mathcal{L}}_{concave}
=
\frac{\operatorname{mean}\max(0,\Delta Q_{a+1}-\Delta Q_a)}
{\operatorname{stopgrad}(\operatorname{median}|\Delta Q|)+\epsilon}.
\]

Log raw/normalized losses, weighted contributions, and preferably gradient norms.

### 11.6 Add a hard support gate

Measure each seed separately. Suggested preregistration candidates:

- p99.5 discounted return below 80% of support magnitude;
- median boundary atom mass below 10%;
- no more than 5% of Q within one atom of a boundary.

These are engineering thresholds requiring instructor approval before new training.

### 11.7 Fix provenance

Phase 2C results must contain source manifest or commit, bundle/notebook/plan hashes, commands, versions, GPU, precision benchmark, input/output hashes, reward scale, support, and every result hash.

---

## 12. Execution sequence

### Stage A — return audit

Implement/unit-test the audit, run on development only, archive raw CSV, then select/freeze scale/support by the declared rule.

### Stage B — seed-2299 smoke

- 20–30 episodes;
- at least one full 25-pair curriculum pass;
- reinitialized categorical heads;
- explicitly warm-started trunk;
- empty replay;
- normalized structural losses;
- FP32 unless benchmark proves BF16 faster.

Gate: finite, no collapse, no boundary pile-up, high-more > high-fewer, ordering improved, QoS viable, no auxiliary domination. Stop on failure.

### Stage C — three seeds

Only after freezing Stage B configuration: run 2299/3299/4299 from corresponding registered trunks for 100–125 episodes. Every seed must pass; do not average away failure.

### Stage D — preregistration addendum

Hash final equations/configuration, warm-start distinction, checkpoints, metrics, censoring, baselines, training-seed aggregation, paired statistics, multiplicity, and no-further-tuning rule.

### Stage E — held-out evaluation

Use all three checkpoints. Do not cherry-pick. For each environment seed, aggregate the three HTA outcomes with a predeclared statistic, then pair with the same baseline seed. Do not treat repeated baselines as 90 independent samples.

Run 3100–3104 pilot first, then 4000–4029 only if the frozen gate passes.

---

## 13. Doubts and unresolved decisions

1. Empirical discounted returns are not archived, so reward scale is unresolved.
2. High future harvest does not imply every immediate slot marginal must always be larger; queue/horizon may justify deferral.
3. Projected-node changes reach only 3.27%; this is eleven times higher but still uncommon.
4. Seed 2299 saturation may reflect initialization, replay, or auxiliary interaction.
5. Lifetime is mixed, not dominant.
6. Death is observed in 62.4–69.6% of episodes and is therefore structurally active; its below-0.5% normalized reward share still raises a calibration question, but not a horizon-reachability question.
7. Node-indexed heads may memorize node/location identity; permutation/identity ablation is needed.
8. Thermal HMM is a simulated default, not real-trace validation.
9. Frozen CH is correct for causal isolation but limits claims.
10. Result provenance is incomplete.
11. Old Phase 4 preregistration is incompatible.
12. Strong competitors must be discussed without invalid numerical cross-paper comparison.

---

## 14. Allowed and disallowed claims

### Allowed

- Phase 2B materially increases decision-level hybrid-trajectory sensitivity on development states.
- The direction replicates across three optimizer lineages.
- Development service, fairness, and delivery improve.
- Lifetime is mixed.
- Seed 2299 is support-saturated and not final.
- The defensible distinction is clustered frozen-CH evaluation plus joint HMM features, shared branching slot counts, and budget projection.

### Disallowed

- Publication-ready model.
- Formal original Phase 2 pass without qualification.
- Held-out/statistical superiority.
- Lifetime dominance.
- First hybrid-EH TDMA.
- First per-node harvest-aware RL scheduler.
- No EH-WSN paper models idle listening.
- Real thermal forecasting validation.
- Treating 3 checkpoints × 30 repeated baseline seeds as 90 independent trials.
- Dropping seed 2299 after observing failure.

---

## 15. Key references

1. Bellemare, Dabney, and Munos, “A Distributional Perspective on Reinforcement Learning,” ICML 2017.  
   <https://proceedings.mlr.press/v70/bellemare17a.html>

2. Rowland et al., “An Analysis of Categorical Distributional Reinforcement Learning,” AISTATS 2018.  
   <https://proceedings.mlr.press/v84/rowland18a.html>

3. Hessel et al., “Rainbow,” AAAI 2018.  
   <https://arxiv.org/abs/1710.02298>

4. Tavakoli et al., “Action Branching Architectures,” AAAI 2018.  
   <https://doi.org/10.1609/aaai.v32i1.11798>

5. Sharma et al., structure-aware EH RL, IEEE TSP 2020.  
   <https://doi.org/10.1109/TSP.2020.2973125>

6. Chen et al., structure-enhanced scheduling DRL, IEEE TWC 2024.  
   <https://doi.org/10.1109/TWC.2023.3277861>

7. Runje and Shankaranarayana, monotonic networks, ICML 2023.  
   <https://proceedings.mlr.press/v202/runje23a.html>

8. Gong et al., SHR-TDMA, 2020.  
   <https://doi.org/10.1049/iet-com.2019.0977>

9. Ge et al., clustered solar-EH cooperative RL, 2021.  
   <https://doi.org/10.1177/15501477211007411>

10. Gong et al., FFSS/AFSS, 2021.  
    <https://doi.org/10.1049/cmu2.12243>

11. Movva et al., S2A2MAC, 2022.  
    <https://doi.org/10.1002/dac.5202>

12. Dutta et al., contextual joint slot/sleep DRL, IEEE TGCN 2024.  
    <https://doi.org/10.1109/TGCN.2024.3358230>

13. Sarang et al., HENO-MAC, 2024.  
    <https://doi.org/10.1109/WCNC57260.2024.10571258>

14. Eriş et al., learned underwater EH MAC, 2024.  
    <https://doi.org/10.3390/s24175791>

Local map: HTA_MAC_2020_2026_EXPANDED_LITERATURE_AND_COMPETITOR_MAP.md.

---

## 16. Exact starting instructions for the next agent

1. Read this report completely.
2. Read the prior Phase 2B report, registered audit, expanded literature map, and old preregistration.
3. Treat downloaded results as immutable evidence.
4. Re-run artifact/finite checks.
5. Implement the discounted-return audit with analytical unit tests.
6. Do not choose scale before the development return CSV exists.
7. Write/hash a Phase 2C mini-plan with support rule and gates.
8. Implement scale/support change plus output-head reinitialization.
9. Normalize auxiliary losses.
10. Run one 20–30 episode seed-2299 smoke.
11. Inspect support, sensitivity, QoS, lifetime, and loss balance.
12. Stop on any failed gate.
13. Only then build a three-seed Colab bundle.
14. Do not access held-out/Phase 4 seeds before the addendum is frozen.

---

## 17. Stop conditions

Stop if any NaN/infinite value appears; always-sleep occurs; boundary mass exceeds the declared gate; returns exceed support; high-more does not exceed high-fewer; auxiliary losses dominate; seed 2299 loses the mechanism; FND loss exceeds tolerance; hashes mismatch; held-out seeds are used during calibration; or anyone proposes excluding seed 2299 after failure.

---

## 18. Final decision

Phase 2B achieved its mechanism repair: the policy reacts to hybrid trajectory features in projected MAC decisions across three optimizer lineages.

It also exposed a separate C51 design error: seed 2299 collapses onto the upper support atom. The confirmation gate did not measure this.

The fastest defensible path is a narrow Phase 2C return-scale repair, beginning with a development-only discounted-return audit and one seed-2299 smoke test. Once the categorical distribution is non-saturated and the mechanism remains active, repeat three-seed confirmation and freeze it for held-out evaluation.
---

## 19. Phase 2C entry-gate addendum — 4 August 2026

This addendum resolves the instructor feedback received after the first report. Where it conflicts with an earlier proposed instruction, this section is authoritative. No Phase 2C training may start until the configuration/mini-plan encoding these decisions is hashed and the source commit is recorded.

### 19.1 Death-event activation — resolved empirically

The 300-step horizon does expose death. The reproducible audit command is:

```powershell
python -B experiments/audit_phase2b_death_activation.py HTA_MAC_Phase2B_Confirmation_Results_20260803 --output-prefix outputs/phase2/phase2b_death_activation_audit
```

| Training seed | Death weight | Death events | Episodes with death | Mean/median/max deaths | First death episode | Last-50 deaths (episodes) |
|---:|---:|---:|---:|---:|---:|---:|
| 2299 | 2.0 | 83 | 78/125 (62.4%) | 0.664 / 1 / 3 | 1 | 30 (29) |
| 3299 | 2.0 | 87 | 84/125 (67.2%) | 0.696 / 1 / 2 | 1 | 33 (32) |
| 4299 | 2.0 | 91 | 87/125 (69.6%) | 0.728 / 1 / 2 | 3 | 39 (37) |

All 375 episodes record `t_fnd`. Therefore the death term is active and no horizon extension is justified by the proposed “zero deaths” failure mode. The instructor feedback assumed `w3=10`, but the immutable Phase 2B summaries and `config/reward_calibration.json` record an effective death weight of 2.0. Restoring 10 would be a reward redesign, not a correction, and would require a fresh return audit and three-seed confirmation. The present Phase 2C repair keeps weight 2.0 frozen so C51 scaling is the only reward-side change.

Archived outputs:

- `outputs/phase2/phase2b_death_activation_audit.json`
- `outputs/phase2/phase2b_death_activation_audit.csv`

### 19.2 One pooled/worst-tail reward scale — locked rule

For each development schedule seed (j\in\{2300,\ldots,2304\}), compute the 99.5th percentile of discounted returns from the complete fixed reference-rollout set. Define (Q^*=\max_j Q_{0.995}(G\mid j)), then compute one (c=\min(1,0.8V_{max}/Q^*)). The maximum—not the mean—protects against the fattest observed development tail.

The following are prohibited:

- per-training-seed scales;
- per-checkpoint scales;
- recomputing `c` after seeing confirmation or held-out results;
- using seeds 3100–3104 or 4000–4029 to select `c`;
- silently changing the reference rollout policy or return horizon.

The locked configuration must contain the five quantiles, (Q^*), `c`, sample counts, return horizon, discount, reference-policy/checkpoint hashes, and raw CSV hash. The same `c` is used for the seed-2299 smoke, rescaling-only ablation, full three-seed confirmation, and later budget arms.

A frozen scale can become stale as the policy improves. If the predeclared boundary-mass gate fails later, the escalation is PopArt-style adaptive target normalization, not repeated manual scale tuning. PopArt is a fallback, not part of the first repair, because adding it now would confound the narrow C51 diagnosis. See van Hasselt et al., “Learning values across many orders of magnitude,” NeurIPS 2016: <https://proceedings.neurips.cc/paper_files/paper/2016/hash/5227b6aaf294f5f027273aebf16015f2-Abstract.html>.

### 19.3 Load-bearing-change ablation — added

Add one 125-episode seed-2299 rescaling-only arm. It must use the identical Phase 2C return scale, C51 support, categorical-head reinitialization, trunk initialization, replay settings, curriculum, schedule seeds, and random seed as the full repaired seed-2299 arm, with only:

```text
trajectory_loss_weight = 0
concavity_loss_weight  = 0
```

This paired arm distinguishes the input/return-scale repair from the two structural auxiliary losses. Report ordering, high-more/high-fewer, projected-node changes, boundary mass, Q-boundary proximity, FND, throughput, delivery, packet Jain fairness, and residual-energy metrics. Do not attribute the 79.63% ordering result to the structural losses unless the paired ablation supports that attribution.

### 19.4 Node-identity shortcut test — added

Use at least 20 deterministic branch permutations per evaluated checkpoint. For a permutation (\pi): move each physical node's full state row, validity mask, queue-feasible cap, and action mask to branch (\pi(i)); run the policy; inverse-permute the selected allocation back to physical node IDs before budget projection/environment stepping. This changes branch identity while preserving the physical state and environment semantics.

Report action-vector agreement, projected allocation disagreement, return, FND, throughput, delivery, packet fairness, residual-energy metrics, and hybrid sensitivity relative to the unpermuted evaluation. If performance collapses, the network is branch-identity-dependent and the paper must not claim node-permutation generalization. For the current fixed 100-node deployment this is a scope limitation, but it is a hard blocker for any transferable “state-conditioned policy” claim.

### 19.5 Energy-based fairness — added now

Packet-delivery Jain fairness is retained but renamed explicitly. Add the following episode-end metrics over alive nodes:

\[
J_E=\frac{\left(\sum_i E_i\right)^2}{n_{alive}\sum_i E_i^2},
\qquad
CV_E=\frac{\sigma(E_i)}{\mu(E_i)}.
\]

Also report mean, minimum, and 10th-percentile residual energy. `J_E` is never interpreted alone because equal near-zero batteries can appear perfectly fair; it must be read together with the energy level and survival outcomes. The same reporting block is required in the Phase 2C smoke, ablation, confirmation, and later Phase 4 table.

### 19.6 Dutta et al. 2024 primary-source differentiation audit — completed

Primary-source evidence was taken from the IEEE record and the author-uploaded full manuscript for DOI [10.1109/TGCN.2024.3358230](https://doi.org/10.1109/TGCN.2024.3358230).

The actual mechanism is not the same as HTA-MAC:

| Dimension | Dutta et al. 2024 | HTA-MAC |
|---|---|---|
| Topology/control | Decentralized arbitrary mesh; localized per-node/per-flow agents | Clustered terrestrial EH-WSN under frozen exogenous CH schedule |
| Slot allocation | Tier-I per-node MAB selects one collision-free transmission slot | Shared Branching Dueling C51 selects a discrete per-node slot-count vector |
| Transmit/sleep action | Tier-II per-flow CDQL selects a discretized transmission probability in the allocated slot | Per-node action is integer slots `0..n_max`; zero is sleep |
| State/context | Queue-length temporal gradient plus estimated traffic-flow-rate context | Residual energy, forecast moments, solar/thermal HMM transition features, queue/history, cluster size |
| Other slots | Tier-III per-node MAB chooses sleep versus awake/listen and explicitly reasons about idle listening | Joint budget projection allocates slot counts and explicit idle energy is charged in the common environment |
| Constraint | Collision avoidance emerges through distributed MAB rewards | Hard frame-budget feasibility through deterministic marginal-Q projection |
| Energy source | Thin fixed energy budget; the inspected manuscript contains no harvest or solar model | Simulated hybrid solar plus synthetic-auxiliary thermal harvesting |

Consequences for claims:

- HTA-MAC cannot claim the first learned joint slot/sleep scheduler, the first contextual DRL MAC, or the first learned mechanism addressing idle listening.
- C3 remains only a simulator/accounting contribution relative to HEART-CH and the paired evaluation environment.
- The defensible distinction is the exact clustered hybrid-HMM, budget-constrained multi-slot-count, shared-branching formulation—not slot/sleep DRL in general.
- Dutta et al. is a prominent qualitative competitor, but a post-hoc numerical baseline is not added to the frozen experiment because its mesh topology, three-tier decentralized control, one-slot collision-learning semantics, traffic context, and receiver model are materially different.
- The authors' 2025 energy-harvesting follow-on remains a separate primary-source audit item before manuscript freeze.

Primary-source mechanism evidence: the author manuscript describes the three-tier architecture and one-slot MAB allocation, its queue-gradient state, flow-rate context, discretized transmit-probability action, and Tier-III idle-listening-aware sleep/listen MAB: <https://www.researchgate.net/publication/377688183_Contextual_Deep_Reinforcement_Learning_for_Flow_and_Energy_Management_in_Wireless_Sensor_and_IoT_Networks>.

### 19.7 Git provenance — hard gate

Create a scoped source/report commit before Phase 2C implementation. Do not add result ZIPs, model directories, downloaded evidence trees, smoke outputs, or unrelated Phase 4 files. Record the branch, commit, clean/dirty state, and exact artifact hashes in the Phase 2C mini-plan.

Verified provenance boundary:

- branch: `codex/phase2b-provenance`;
- Phase 2B implementation/evidence commit: `d489f988cd458441a94a92b8f64dbb6396f25f2f`;
- validation: `python -B -m compileall -q agents experiments validation` passed;
- validation: `python -B -m pytest validation -q -p no:cacheprovider` → `43 passed, 93 warnings` in 44.27 s;
- warnings are third-party Torch/Torch-Geometric deprecations, not test failures;
- large ZIPs, downloaded result trees, checkpoints, local smoke directories, and unrelated `outputs/phase4/` remain uncommitted;
- the only residual untracked path after the implementation commit is the pre-existing `outputs/phase4/` tree, deliberately outside this provenance scope.

This report-only closure is committed separately so the implementation/evidence commit can be named without a circular self-hash.

### 19.8 Revised Phase 2C execution order

1. Commit the Phase 2B source, audits, reports, and confirmation notebook/runner with large artifacts excluded.
2. Implement and test the fixed reference-return audit.
3. Compute and hash the five development-seed quantiles, worst-tail (Q^*), and one frozen `c`.
4. Add residual-energy metrics and the branch-permutation evaluator with unit tests.
5. Freeze/hash the Phase 2C mini-plan and configuration.
6. Run the short seed-2299 full-repair smoke; stop on support, mechanism, or energy-fairness failure.
7. Run the 125-episode seed-2299 rescaling-only paired arm.
8. Only after attribution and support gates pass, run the full three-seed confirmation.
9. Keep held-out seeds untouched until the preregistration addendum is frozen.

No model is publication-ready at this boundary. Phase 2B established development-set mechanism response and exposed output saturation; Phase 2C must establish numerical validity, attribution, identity robustness, and energy-distribution behavior before held-out evaluation.
---

## 20. Second Phase 2C gate clarification — death salience, curriculum shift, return reference, and Dutta 2025

This section answers the second instructor review. It supersedes any statement in Section 19 that calls the death calibration, reference-return policy, or competitor search closed without the qualifications below.

### 20.1 Death signal: active and reward-scale-salient, but causal learning remains unproven

`experiments/audit_phase2b_death_activation.py` now compares the death penalty at firing timesteps with the packet-delivery term on the timestep scale.

| Seed | Logged steps | Death-firing steps | Firing-step fraction | Mean weighted death magnitude when firing | Mean weighted packet term per logged step | Death/packet ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 2299 | 18,978 | 78 | 0.411% | 2.1282 | 1.3644 | 1.5598 |
| 3299 | 19,342 | 84 | 0.434% | 2.0714 | 1.5010 | 1.3800 |
| 4299 | 19,205 | 87 | 0.453% | 2.0920 | 1.4938 | 1.4005 |

A single death contributes exactly `-2.0`; averages exceed 2 because a few terminal steps contain two or three simultaneous target-cluster deaths. The death term is therefore approximately 1.38–1.56 times the typical packet-delivery contribution when it fires, not 1/200 of it. This rejects the specific numerical-drowning hypothesis and supports keeping `w3=2.0` frozen during the narrow C51 scaling repair.

Evidence boundary: Phase 2B did not archive a per-step trace separating packet reward on death versus non-death timesteps. The denominator above is the exact mean packet term over all logged timesteps. Death-firing steps are only 0.41–0.45% of all steps, so this is a close proxy for a typical non-death timestep, but it is not falsely labeled an exact non-death-only mean. Reward-scale salience also does not prove that the learned network assigns correct causal credit to death or that lifetime behavior generalizes.

### 20.2 The death audit measures the local curriculum, not full network-wide policy evaluation

Section 19.1 audits `DynamicClusterTrainingEnv` episodes. The underlying `ScheduledIntraClusterMACEnv`, HEART-CH schedules, 0.5-J initial energy, radio model, HMM assets, queue rules, full-slot idle model, and harvest update are shared with Phase 3. There is no smaller battery or accelerated physical depletion constant.

The rollout semantics differ materially:

| Dimension | Phase 2B curriculum | Phase 3 held-out pilot |
|---|---|---|
| Horizon | At most 300 scheduled rounds | 1,633–1,730 rounds before schedule censoring |
| Policy coverage | Learned policy controls one scheduled target cluster; all other clusters use static-equal TDMA | Selected policy controls every cluster |
| Exploration | Epsilon-greedy, 0.10→0.03 | Greedy/deterministic policy evaluation |
| Budget/model | Repaired shared-branching budget-12 checkpoints | Earlier authoritative budget-8 checkpoint |
| Episode terminal | Target member/CH death, global termination, or truncation | Full-network termination or schedule exhaustion |
| Death reward | Newly dead target members plus target CH | No training reward; FND is a network metric |

The zero-FND statement in Phase 3 applies only to the earlier greedy HTA budget-8 policy. The same full-network environment produced FND events for every conventional baseline (approximately 110–207 rounds in the five-seed pilot). Therefore the environment is not incapable of death; the HTA policy slept sufficiently often to remain right-censored.

The distribution gap was not introduced as a deliberately calibrated accelerated-stress curriculum. It arose from the local-credit training wrapper: applying a candidate to one target cluster while retaining static TDMA elsewhere made single-cluster learning tractable and preserved global node identity. It is now treated as an explicit curriculum/evaluation shift. Consequences:

- do not use curriculum FND as evidence of network-wide lifetime;
- do not interpret the death-term audit as representative deployment event frequency;
- add a development-only network-wide greedy rollout for every Phase 2C checkpoint before confirmation acceptance;
- report both local-curriculum and network-wide outcomes;
- if the network-wide policy remains completely censored while the local curriculum dies near 150 rounds, describe the curriculum as a stress/credit-assignment surrogate only after empirically quantifying that gap—not by design intent.

### 20.3 Exact reference policies for the return-scale audit

The earlier phrase “fixed reference-rollout set” was underspecified. No `c` has been computed yet, and no pre-repair registered checkpoint is authorized as the sole reference.

The reference checkpoint set is the complete repaired Phase 2B confirmation trio:

| Optimizer seed | Checkpoint | SHA-256 |
|---:|---|---|
| 2299 | `HTA_MAC_Phase2B_Confirmation_Results_20260803/runs/phase2b_confirm_shared_b12_seed2299_125ep/branching_c51.pt` | `F67962F4F48871D7A7BA9446F1E528A6AE381305CED4E9207DB68551392C8049` |
| 3299 | `HTA_MAC_Phase2B_Confirmation_Results_20260803/runs/phase2b_confirm_shared_b12_seed3299_125ep/branching_c51.pt` | `B1CFBC377BE1A5BAE6FE0D571CD3AF7DE06522A0C7DA45CB4A8888376632F182` |
| 4299 | `HTA_MAC_Phase2B_Confirmation_Results_20260803/runs/phase2b_confirm_shared_b12_seed4299_125ep/branching_c51.pt` | `350B443B3CE84F47A2DC6E6377D86C08010C8375091E6610E33019E0A5C6DE61` |

For every checkpoint (k), run the complete 25-pair development curriculum formed by schedule seeds 2300–2304 and their five initial target ranks, using the frozen budget-12 policy with the predeclared deterministic audit action seed and `epsilon=0.10`, the Phase 2C starting exploration rate. Archive every raw per-step reward and return. The locked tail statistic becomes

\[
Q^*=\max_{k\in\{2299,3299,4299\}}\max_{j\in\{2300,\ldots,2304\}}Q_{0.995}(G\mid k,j),
\qquad
c=\min\left(1,\frac{0.8V_{max}}{Q^*}\right).
\]

This is conservative across both schedule and optimizer lineage. The run manifest must additionally hash the action RNG seed, environment/schedule manifest, reward configuration, horizon, discount, and raw transition CSV. If any checkpoint hash differs, scale selection aborts.

### 20.4 Rescaling-only attribution now requires two lineages

The rescaling-only ablation remains required for saturated seed 2299 and is extended to non-saturated seed 3299. Both arms run 125 episodes with identical initialization, support, scale, action seeds, curriculum order, and optimizer settings as their corresponding full Phase 2C arms, with only the trajectory and concavity weights set to zero.

Seed 2299 diagnoses recovery from the worst saturation case; seed 3299 checks that attribution is not an artifact of that outlier. A mechanism statement requires directionally consistent paired evidence across both. With two lineages the result remains an ablation replication, not a precise population-level percentage decomposition. No paper sentence may claim “X% due to scaling and Y% due to auxiliary losses” from these two runs alone.

### 20.5 Targeted identity swaps added to the permutation test

Retain 20 deterministic random active-branch permutations per checkpoint. Add at least 10 targeted swaps per checkpoint that pair:

- one alive backlog-eligible high-harvest node (`S6` or `S8` solar taxonomy), and
- one alive backlog-eligible declining/low node (`S1`, `S4`, or `S7`),

preferably within the same cluster, with equal queue-derived action cap and residual energy matched within a predeclared tolerance. Move the complete state row, mask, cap, and action mask; inverse-map actions before projection and environment stepping.

Report whether the allocation follows the moved feature bundle or stays associated with the original branch head, plus marginal-Q ordering and projected allocation. Random permutations test general branch dependence; targeted high/low swaps are the sensitive state-versus-identity diagnostic.

### 20.6 Dutta et al. 2025 primary-source audit — now a top-tier competitor

The follow-on is H. Dutta, A. K. Bhuyan, and S. Biswas, “Cooperative Reinforcement Learning for Energy Management in Multi-Hop Networks With Energy Harvesting,” *IEEE TGCN*, vol. 9, no. 4, pp. 1783–1793, 2025, DOI [10.1109/TGCN.2025.3544073](https://doi.org/10.1109/TGCN.2025.3544073).

Primary-source findings:

- multi-hop, decentralized EH sensor/IoT network;
- two-state Markov solar process generates high/low radiation energy arrivals;
- two tabular Q-learning agents per node share a reward;
- the transmission agent observes quantized harvested-energy influx and chooses discretized transmission probability;
- the sleep agent observes the transmission policy and chooses discretized transceiver-on probability;
- a neighbor-shared learning-confidence parameter suppresses unreliable downstream updates;
- no hybrid solar-plus-thermal model, no clustered frozen-CH evaluation, no learned per-node multi-slot-count vector, no Branching DQN, and no hard cluster slot-budget projection.

This paper invalidates broad novelty for Markov-solar-conditioned per-node RL transmit/sleep scheduling. It is more urgent for positioning than Dutta et al. 2024 and must appear prominently in the introduction/related-work competitor table. The remaining defensible intersection is hybrid solar/thermal state-transition features plus a shared branching categorical value network for hard-budgeted intra-cluster discrete multi-slot counts.

Detailed durable audit: `HTA_MAC_DUTTA_2025_PRIMARY_SOURCE_AUDIT_20260804.md`.

### 20.7 Revised immediate decision

Before Phase 2C training:

1. keep `w3=2.0` frozen; the timestep audit rejects numerical drowning but not causal-learning uncertainty;
2. implement the per-step reference-return logger using all three repaired checkpoint hashes and all 25 development pairs;
3. compute one worst-lineage/worst-schedule `Q*` and freeze/hash `c`;
4. add the network-wide development rollout gate, two-lineage rescaling-only ablation, random permutations, targeted swaps, and residual-energy metrics to the mini-plan;
5. keep held-out seeds untouched;
6. do not start confirmation training until the mini-plan and implementation tests pass.

The C51 repair remains necessary. The literature audit and curriculum disclosure narrow the claims but do not eliminate the exact HTA-MAC formulation as a potentially publishable contribution.