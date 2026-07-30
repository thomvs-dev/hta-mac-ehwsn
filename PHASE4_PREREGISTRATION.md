# HTA-MAC Phase 4 Preregistration and Analysis Freeze

**Frozen:** 2026-07-30 (Asia/Calcutta)  
**Status:** Binding analysis plan written before the schema-v2 authoritative budget sweep and before any Phase 4 test-seed evaluation.

## 1. Scope and evidence boundary

HTA-MAC changes only intra-cluster MAC slot allocation. HEART-CH cluster-head (CH) selection remains an exogenous, frozen per-round schedule. Routing and CH-policy retraining are outside scope. Schedule schema v2 seeds Python, NumPy, and PyTorch before frozen-policy inference and right-censors evaluation when the inherited HEART-CH episode terminates. A schedule is never extended by stale-frame replay.

The historical schema-v1 budget-8 pilot is development evidence only. It cannot select a Phase 4 result, claim, test, checkpoint, or hyperparameter.

> Absolute round counts in this paper are not comparable to HEART-CH Table V because this work introduces explicit idle-listening accounting absent from the original energy model; all comparisons here are internally consistent across the seven policies evaluated under identical accounting.

The quoted seven-policy sentence describes the original single-budget comparison. The registered budget sweep expands HTA-MAC into five resource-budget arms while retaining the same six comparison/control policies.

## 2. Frozen configurations

### 2.1 HTA-MAC arms

The primary architecture is a shared global decision module followed by 100 node-indexed dueling action heads; the active-node mask and zero padding preserve the fixed global node identity. Projection budgets are **{8, 12, 16, 20, 24} slots per cluster per round**. Each budget is trained independently from a random initialization under optimizer/training seeds **{2299, 3299, 4299}**. All runs use development schedule seeds **{2300, 2301, 2302, 2303, 2304}**, 500 episodes, a maximum of 300 environment steps per episode, schedule schema v2, and the same frozen reward weights and agent architecture. No budget arm is warm-started from another arm or from the historical schema-v1 checkpoint.

A trained replicate is admissible only if its recorded Phase 2 gate passes: no non-finite value, no always-sleep collapse, no reward term above the frozen domination threshold, differentiated trajectory Q-values, reward convergence, and deterministic greedy-policy stability across at least three tail checkpoints. Failed runs remain archived and are not silently replaced. Any rerun uses the identical registered training seed/configuration and is identified as a technical rerun.

If one or more of the three registered replicates for a budget fails its gate, that budget is reported as training-unstable. It is not promoted as a successful Phase 4 arm. No new seed is substituted.

### 2.2 Architecture ablation

The independent-DQN ablation uses 100 separately parameterized local dueling C51 networks, no shared global decision module, the same budget projection, and the same state/reward/training protocol. It is trained at the fixed middle budget **12** under training seeds **{2299, 3299, 4299}**. Its Phase 4 comparison with the budget-12 shared-branching arm is secondary and uses the same 30 paired environment seeds and seed-level aggregation rule. Parameter counts, runtime, training-gate outcomes, and endpoint effects are reported; a failed ablation training seed is retained as an instability result rather than replaced.

### 2.3 Baselines and control

The comparison arms are static-equal TDMA, tuned energy-proportional, tuned harvest-proportional, tuned S2A2MAC adaptation, tuned FFSS adaptation, and the random-budgeted stochastic floor. Baseline parameters are frozen from schema-v2 development seeds **{2500, 2501, 2502, 2503, 2504}** in `config/phase3.yaml`. Phase 4 test seeds cannot retune them. The random policy is a formal stochastic lower-bound comparator, not a literature baseline.

### 2.4 Test seeds and horizon

Phase 4 uses exactly **30 paired environment seeds, 4000 through 4029 inclusive**, with requested horizon 3000 rounds. Each policy sees the same frozen schedule and exogenous stochastic trajectory for a given seed. Schedule termination creates right censoring and is not a crash or exclusion.

For each HTA-MAC budget and test seed, the metric used in paired policy tests is the arithmetic mean across its three independently trained, gate-passing replicates. The three replicate values and their dispersion are also reported. This keeps the inferential unit as the environment seed (n=30) and avoids treating three networks on the same trajectory as independent observations.

## 3. Endpoints

### 3.1 Co-primary endpoints

1. **Common-horizon restricted FND-free time:** for every policy-seed observation, use `min(T_FND, tau)`, where an unobserved FND uses its censor time and `tau` is the minimum valid observation horizon across every registered Phase 4 arm and seed. Higher is better. Kaplan-Meier event counts and median, when estimable, are additionally reported.
2. **Packet delivery ratio:** total packets delivered divided by total packets generated, including the initialized queue population in the denominator. Higher is better.

These endpoints represent the lifetime/QoS Pareto tradeoff. No single scalar utility is constructed after seeing results. A budget is Pareto-dominated if another registered budget is at least as good on both median co-primary endpoints and strictly better on at least one.

### 3.2 Secondary endpoints

T_HND (with the same censor-aware treatment), throughput, idle-listening energy, stale-drop ratio, queue-service Jain fairness, residual-energy Jain fairness over all nodes including zeros, residual-energy coefficient of variation, mean/minimum residual energy, total consumed energy, packets/J, and stale/death/overflow drop counts.

## 4. Confirmatory comparisons and multiplicity

For each of five HTA-MAC budgets, each co-primary endpoint is compared against (a) static-equal TDMA and (b) the tuned S2A2MAC adaptation using a two-sided paired Wilcoxon signed-rank test. This creates one confirmatory family of 20 p-values (5 budgets x 2 endpoints x 2 baselines), corrected by Holm's method at family-wise alpha 0.05. A claim of difference requires its Holm-adjusted p-value below 0.05; the effect direction must be stated from the observed paired shift.

All other policy/endpoint tests, including comparisons with the random floor, are secondary. They are reported with raw two-sided paired Wilcoxon p-values and a separate Holm adjustment across the complete secondary family. No one-sided test is introduced after results are observed.

Every paired comparison reports:

- sample count and event/censor count where applicable;
- median and IQR per arm;
- median paired difference (HTA-MAC minus comparator);
- paired Hodges-Lehmann shift, computed as the median Walsh average of paired differences;
- matched-pairs rank-biserial correlation, signed as HTA-MAC minus comparator;
- wins, ties, and losses;
- raw and applicable Holm-adjusted p-values.

For lower-is-better endpoints, a negative HTA-minus-comparator shift favors HTA-MAC. For higher-is-better endpoints, a positive shift favors HTA-MAC. When every paired difference is zero, the statistic is recorded as undefined, p=1, and rank-biserial effect=0.

## 5. Missing data, failures, and integrity rules

- Every scheduled run must produce a raw per-trial row. No NaN or infinity is accepted.
- A software/hardware interruption may be rerun only with the identical policy, checkpoint, seed, and configuration; both the failure log and completed rerun are archived.
- A scientific failure (training collapse, gate failure, node death, or schedule censoring) is never removed or replaced.
- Policy exceptions, missing rows, seed mismatch, hash mismatch, or unregistered checkpoint use fail the Phase 4 gate.
- All checkpoints, schedule caches, raw CSVs, summaries, and preregistration files enter the artifact manifest with SHA-256 and byte size. `validation/verify_manifest.py` must print zero failures before results are used.

## 6. Decision and claim rules

The paper reports the complete five-budget Pareto frontier, including negative and mixed results. A budget may be highlighted only by a rule declared here: it is non-dominated on the two median co-primary endpoints, all three training replicates pass, and its delivery ratio is not lower than the random floor's median. If multiple budgets meet this rule, all remain highlighted; no post hoc single winner is selected.

No superiority claim is made from median direction alone. No numerical comparison is made against a publication using materially different network geometry, radio range, energy accounting, or traffic semantics. The thermal auxiliary is described as synthetic upstream-default parameterization, not a dataset-trained thermal HMM.

## 7. Frozen taxonomy and terminology

The reward taxonomy follows the upstream HEART-CH solar-state labels exactly: one-based **S6 and S8** are high-harvest/good states; **S1, S4, and S7** are avoid/critical declining states. Code stores these as zero-based `{5,7}` and `{0,3,6}`. No categorical thermal reward label is invented because the thermal auxiliary has no validated taxonomy. Observation features are called **state-conditioned transition probabilities**, not Bayesian posteriors.

## 8. Amendments

Any amendment must be dated, hashed separately, explain why it was necessary, and be written before the affected result is inspected. The original frozen file and hash remain archived. Amendments cannot erase failed or unfavorable results.