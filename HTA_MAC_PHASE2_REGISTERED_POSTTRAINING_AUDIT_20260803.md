# HTA-MAC Phase 2 Registered Sweep: Post-Training Audit and Decision

**Audit date:** 3 August 2026  
**Artifact audited:** `HTA_MAC_Phase2_Registered/`  
**Source revision recorded by the artifact:** `704ad5a5aaf4764ec57a7f74c4d8b6f26005b4b7`  
**Decision:** computational training is complete and internally valid, but the
scientific Phase 2 trajectory-response gate is **not yet established**. Do not
describe the current checkpoints as publication-ready HTA-MAC models and do
not use the existing numerical gate alone to support a trajectory-conditioned
policy claim.

## 1. Executive finding

The Colab campaign completed exactly the registered 18-run matrix:

- shared Branching Dueling C51 at budgets 8, 12, 16, 20, and 24;
- three optimizer seeds per budget: 2299, 3299, and 4299;
- independent-DQN ablation at budget 12 with the same three seeds;
- 500 training episodes per run over the five-seed, 25-cluster development
  curriculum, with a maximum of 300 steps per episode.

All 18 runs pass the frozen implementation checks: no non-finite values, no
always-sleep collapse, convergence within the declared tolerance, stable tail
snapshots, balanced logged reward terms, and a nonzero S1-versus-S8 Q-value
difference. The artifact is complete and hash-valid.

However, the gate's trajectory test only required an absolute Q difference
greater than `1e-4`. The original single-state probes changed the greedy local
action in **0 of 18 runs**. A separate post-hoc audit across 98 active-node
counterfactuals per checkpoint found only 8 local argmax changes in 1,764
probes, 3 changes to the probed node's projected slot count, and 20 changes to
the full projected action vector. In no probe did the S8 counterfactual give
the probed node more slots; the three node-level changes gave it fewer slots.

The correct interpretation is therefore:

> The trained networks are numerically sensitive to the solar trajectory
> features, but current evidence does not show material or directionally
> correct per-node slot differentiation caused by those features.

That distinction matters because trajectory-conditioned differentiation is
the central scientific mechanism, not an optional diagnostic.

## 2. Artifact provenance and integrity

The downloaded folder was audited read-only. The reusable integrity audit is
`experiments/audit_downloaded_phase2.py`; its machine-readable outputs are:

- `outputs/phase2/downloaded_registered_sweep_audit.csv`
- `outputs/phase2/downloaded_registered_sweep_audit.json`

Results:

| Check | Result |
|---|---:|
| Registered runs | 18/18 |
| Manifest-listed run files checked | 126 |
| Missing files | 0 |
| Byte-size mismatches | 0 |
| SHA-256 mismatches | 0 |
| Manifest SHA-256 | `87bcdf907c4c435a807654b23d3b02bc866b5a2c165f5d47c215327df4a97ebb` |
| Registry completion flag | `true` |
| Registered gate-pass count | 18/18 |
| Recorded accelerator | NVIDIA L4 / CUDA |

The older local manifest was created on 1 August 2026 and lists 108 run files;
the downloaded Colab manifest was created on 3 August 2026 and lists 126. All
126 keys in the union of comparable run artifacts differ by presence or hash.
Accordingly, the August 3 Colab artifact is the authoritative Phase 2 sweep.
The older local outputs must not be pooled with it.

The downloaded artifact itself was not edited. All audit products were written
under `outputs/phase2/`.

## 3. Training health

The campaign consumed 18.680 recorded GPU-hours. Median runtime was 42.63
minutes per run (IQR 40.14-49.82 minutes). Across the 18 checkpoints:

- convergence relative change ranged from 0.20% to 3.37%, below the 10% gate;
- the largest recorded FND/throughput/fairness tail-snapshot span was 6.00%,
  below the 10% stability gate;
- no run produced non-finite values;
- no run collapsed to the all-sleep action;
- the largest contribution of any one logged reward term in any run was
  61.46%, below the 80% dominance threshold.

Median reward contribution fractions across runs were 58.46% packet delivery,
11.66% idle energy, 0.51% deaths, 6.33% high-harvest alignment, 9.24% declining
allocation, and 13.62% queue fairness. Thus, the reward logging and balancing
gate is genuinely satisfied; the trajectory-response problem cannot be
dismissed as one term exceeding the declared dominance ceiling.

Passing these checks establishes optimization stability. It does not establish
generalization, superiority over baselines, or use of the intended trajectory
features.

## 4. Development-curriculum performance

The following values are medians over only three optimizer seeds. They come
from greedy evaluation on the development curriculum, not from the 30 paired
Phase 4 test seeds, and must not be reported as final paper results.

| Architecture | Budget | Reward | Packets/step | FND-free steps | Throughput | Queue fairness | Delivery ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| Shared branching | 8 | 154.784 | 6.204 | 148.56 | 10,813.88 | 0.398 | 0.659 |
| Shared branching | 12 | 149.328 | 7.985 | 141.32 | 10,409.48 | 0.438 | 0.674 |
| Shared branching | 16 | 151.157 | 9.496 | 137.08 | 10,271.52 | 0.540 | 0.687 |
| Shared branching | 20 | 145.012 | 10.672 | 134.64 | 10,005.68 | 0.527 | 0.698 |
| Shared branching | 24 | 133.940 | 11.320 | 131.52 | 9,768.64 | 0.554 | 0.702 |
| Independent DQNs | 12 | 139.349 | 8.068 | 141.96 | 10,300.32 | 0.406 | 0.673 |

There is a clear resource/lifetime/QoS frontier. Larger budgets improve
packets per step and delivery ratio, while FND-free time and total throughput
fall. Budget 8 gives the strongest development FND-free time and throughput
but the weakest delivery/fairness values. Budget 24 has the strongest delivery
ratio and median fairness but the weakest FND-free time and throughput.

No single budget should be selected from this table. The preregistration
correctly requires the complete five-budget Pareto frontier on held-out paired
seeds and prohibits a post-hoc winner.

## 5. Shared branching versus independent learners

The shared model contains 2,842,811 trainable parameters; the independent
ablation contains 5,622,700. At budget 12, median training runtime was 43.79
minutes for the shared model and 159.50 minutes for independent DQNs. The
shared design therefore uses 49.44% fewer parameters and trained about 3.64
times faster in this campaign.

Performance evidence is mixed across the three paired optimizer seeds. Shared
minus independent differences at budget 12 were:

| Seed | Reward | FND-free steps | Throughput | Fairness | Delivery ratio |
|---:|---:|---:|---:|---:|---:|
| 2299 | +3.688 | +1.80 | +252.08 | +0.0250 | +0.0003 |
| 3299 | +21.703 | +2.72 | +252.68 | +0.0642 | +0.0041 |
| 4299 | -4.746 | -0.72 | -266.00 | +0.0282 | -0.0047 |

It is defensible now to claim computational efficiency of the shared
architecture for this implementation. It is not defensible to claim superior
network performance from three development seeds. That requires the declared
30-seed paired architecture ablation.

## 6. Recent discovery: the trajectory gate is too weak

### 6.1 Frozen gate behavior

`experiments/train_phase2_fixed_cluster.py::trajectory_q_check` constructs S1
and S8 solar counterfactuals for one active node at equal normalized energy,
updates the solar transition vector and rectified forecast moments, and marks
the run differentiated when:

```text
max_absolute_Q_difference > 1e-4
```

Across the 18 registered runs, median absolute Q difference was 0.00345 (IQR
0.00168-0.00962), but median difference relative to Q magnitude was only
0.000699, or 0.0699% (IQR 0.0160%-0.1040%). The local greedy action was
unchanged in 18/18 single-state checks.

The registry truthfully records that all runs pass the implemented criterion.
The implemented criterion is simply not strong enough to establish the
blueprint's wording that S8 and S1 nodes are visibly differentiated.

### 6.2 Stronger post-hoc diagnostic

`experiments/audit_phase2_trajectory_sensitivity.py` evaluated the immutable
downloaded checkpoints over the original development curriculum. It sampled
98 active node/environment pairs per checkpoint and changed only the probed
node's energy-normalized S1/S8 trajectory block while holding node identity,
mask, and the rest of the global state fixed. Output is archived at:

`outputs/phase2/posthoc_registered_trajectory_policy_sensitivity.json`.

| Test | Count | Fraction |
|---|---:|---:|
| Local branch argmax changed | 8/1,764 | 0.45% |
| Probed node's projected slot count changed | 3/1,764 | 0.17% |
| Full projected action vector changed | 20/1,764 | 1.13% |
| S8 gave probed node more slots | 0/1,764 | 0.00% |
| S8 gave probed node fewer slots | 3/1,764 | 0.17% |

For the 15 shared checkpoints alone, there were 4 local argmax changes in
1,470 probes, 3 probed-node allocation changes, and 20 joint-vector changes.
For the three independent checkpoints, there were 4 local changes in 294
probes and no projected allocation changes.

The shared model can change other nodes' allocations when one node's
trajectory features change, which is consistent with a coupled global trunk.
That is not equivalent to the desired directional per-node response.

This diagnostic is post hoc, uses development states, and samples reset-state
counterfactuals whose initial queue caps often limit nodes to one slot. It is
therefore a diagnostic warning, not a new preregistered hypothesis test. Those
limitations could suppress action changes, but they do not rescue the current
claim: a publication claim requires positive evidence, not an explanation for
why a diagnostic may lack it.

## 7. Likely causes to test, not assume

The available evidence supports several hypotheses but does not identify one
proven cause:

1. **Trajectory features may be behaviorally ignored.** The shared network's
   median counterfactual Q response is much smaller than its Q scale.
2. **Packet, queue, identity, and energy signals may dominate decisions even
   though no reward term violates the 80% balance ceiling.** A ceiling prevents
   pathological domination; it does not guarantee feature utilization.
3. **Discrete actions and queue caps can hide small Q differences.** A change
   can exist without crossing an argmax or marginal-gain boundary.
4. **The categorical shaping reward uses only the current solar state.** The
   synthetic thermal channel has no validated high/declining taxonomy and no
   direct categorical alignment reward. Therefore the present gate cannot
   establish use of the hybrid decomposition.
5. **The single S1/S8 gate is too narrow.** It does not test intermediate
   states, thermal-only counterfactuals, hybrid-versus-solar ablation, or
   backlog/scarcity states where multi-slot actions are feasible.

These are ordered diagnostic targets, not excuses and not established facts.

## 8. Corrective decision before Phase 4

Phase 4 confirmatory evaluation should be held until a development-only
trajectory-use repair is completed. The raw Colab models and registry must
remain unchanged. The minimal defensible sequence is:

1. Extend the post-hoc diagnostic to observed mid-episode states spanning
   queue caps 1-3, scarce and non-scarce clusters, all eight solar states, and
   thermal-only and hybrid counterfactuals.
2. Log marginal-Q slot gains and rank changes, not only raw Q differences.
   Budget projection is driven by marginal gains, so this is the correct
   decision-level quantity.
3. Run feature-block permutation/zeroing on development episodes for solar
   transitions, thermal transitions, harvest moments, energy, queue, previous
   slots, cluster fraction, and ST-GCN embedding. Measure changes in projected
   actions and return.
4. Define an action-level development gate before inspecting a repaired
   model—for example, a material projected-allocation response rate with a
   prespecified directional test. The threshold must be justified and frozen,
   not chosen after seeing the next training run.
5. If the trajectory block remains inactive, rebalance the trajectory reward
   terms or add an auxiliary representation objective, then rerun the full
   registered 18-arm Phase 2 matrix under a new versioned registry. Do not
   replace or relabel the present artifact.
6. Only after that gate passes should a fresh Phase 4 campaign run on paired
   seeds 4000-4029.

The partially completed local Phase 4 folder
`outputs/phase4/registered_30seed_schema_v2/` contains only 334 raw rows from an
interrupted earlier campaign. It is incomplete and must remain quarantined
from the future authoritative Phase 4 result set.

## 9. Consequence of the expanded 2020-2026 literature search

The recent competitor search materially narrows the paper's novelty horizon:

- SHR-TDMA (2020) already combines hybrid energy harvesting and per-node TDMA
  assignment.
- Ge, Nan, and Guo (2021) already use cooperative per-node Q-learning/SARSA in
  a clustered solar EH-WSN with predicted next-slot harvest.
- FFSS/AFSS (2021) already performs forecast-aware slot assignment.
- S2A2MAC (2022) already uses HMM-informed node active periods in clustered
  EH-WSNs; it must not be described as one cluster-wide rule.
- Dutta et al. (IEEE TGCN 2024) already study deep-RL joint slot allocation and
  transmit-sleep scheduling.
- HENO-MAC (2024) already studies a hybrid-source energy-neutral MAC.
- D2PG (2025 issue) already applies deep control in a clustered EH-WSN.
- Dutta et al. (IEEE TGCN 2025) already study cooperative learned
  transmit-sleep schedules in EH networks.
- Seifullaev et al. (IEEE TGCN 2024) already combine Markov harvest dynamics,
  Bayesian inference, and learned transmission policies.

Therefore, the paper must not claim first use of RL, clustering, adaptive TDMA,
hybrid harvesting, prediction-aware control, or learned sleep/slot scheduling.
The current narrow candidate positioning is:

> Among the audited works, HTA-MAC evaluates a clustered terrestrial EH-WSN
> scheduler combining per-node solar-and-thermal HMM state-transition features
> with budget-constrained discrete multi-slot allocation through a shared
> Branching Dueling value network under a frozen exogenous CH schedule.

This is presently a system-description distinction, not a validated
contribution, because the trained policy has not yet demonstrated material
trajectory-conditioned action differentiation. An unrestricted “first” claim
still requires a systematic database search.

The expanded source map is
`HTA_MAC_2020_2026_EXPANDED_LITERATURE_AND_COMPETITOR_MAP.md`. The older
`NOVELTY_AND_CLOSEST_WORK_AUDIT.md` remains useful for its S2A2MAC correction,
but the expanded map is authoritative wherever their coverage differs.

## 10. What can and cannot be claimed now

### Supported now

- The registered Colab training matrix completed without corruption.
- All 18 runs satisfy the exact frozen computational gate implementation.
- Optimization did not collapse to always-sleep.
- The reward accounting is logged and no term crosses the declared 80%
  dominance threshold.
- A development lifetime/delivery/fairness frontier exists across budgets.
- The shared architecture is substantially smaller and faster than 100
  independent networks in this implementation.

### Not supported now

- Publication readiness.
- Held-out superiority over static TDMA, S2A2MAC-style, or FFSS-style.
- Generalization to 30 paired Phase 4 seeds.
- A single best slot budget.
- Performance superiority of shared branching over independent learners.
- Material or directionally correct trajectory-conditioned action selection.
- Usefulness of the thermal channel.
- A universal first-ever novelty claim.
- The planned strict lifetime-dominance contribution or 95% analytical-bound
  validation, which belongs to Phase 5.

## 11. Final decision

The training run was not wasted. It produced a complete, reproducible 18-model
artifact, verified the computational pipeline, exposed a real Pareto frontier,
and demonstrated a strong computational advantage for shared branching. More
importantly, the audit caught a weak scientific gate before it could become a
paper claim.

The correct project status is:

> **Phase 2A complete: registered optimization and systems gate passed. Phase
> 2B blocked: policy-level trajectory-use gate requires repair and retraining
> evidence. Phase 4 confirmatory evaluation is not yet authorized.**

This preserves the work while preventing a numerically “passed” but
mechanistically unsupported model from being labeled publication-ready.
