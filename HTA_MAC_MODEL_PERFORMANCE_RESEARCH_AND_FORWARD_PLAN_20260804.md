# HTA-MAC Model Performance Research, Diagnosis, and Forward Plan

**Prepared:** 4 August 2026  
**Repository:** `F:\WSN\matlab\stage2\hta-mac`  
**Branch:** `codex/phase2b-provenance`  
**Evidence HEAD:** `02c079ce6ee1eca19b0d23b750c92a01164a85eb`  
**Status:** Development diagnosis and implementation plan; not publication evidence

---

## 1. Executive decision

The next useful step is **not** to continue training the current `GlobalBranchingDuelingC51`. Its Phase 2C identity audit has already falsified a required property: allocations change when the same physical node bundles are permuted. The current network flattens the 100-node tensor and uses a different advantage head for each array position. It therefore learns branch position, not only node state.

The recommended repair is a new **permutation-equivariant set-branching C51 policy**, trained from a fresh initialization, with three changes made in a controlled sequence:

1. expose packet-age/expiry state that the environment currently tracks but the policy cannot observe;
2. replace the flattened, node-indexed network with shared per-node encoding, invariant set aggregation, and a shared per-node action head;
3. after that architecture passes identity and learning gates, express delivery/freshness as an explicit development constraint instead of continuing to tune an indirect scalar reward.

This order matters. Changing architecture, state, reward, and distributional algorithm together would destroy attribution. C51 should initially be retained with the already frozen scale and support. PopArt-style adaptive target normalization is the first preregistered escalation if support occupancy remains unstable; IQN is a later, separate alternative rather than an immediate rewrite.

The bounded scientific scope remains unchanged: frozen exogenous HEART-CH cluster schedules, intra-cluster MAC slot allocation only, no CH retraining, no routing, no Pointer Network, and no mixing with HERMES.

---

## 2. Evidence and verification boundary

This report combines four evidence classes:

- direct inspection of the present repository, implementation, archived metrics, and handoffs;
- the completed Phase 2C branch-identity artifact;
- primary research papers and publisher/DOI records collected on 4 August 2026 with Firecrawl;
- clearly labelled engineering inferences that still require implementation and measurement.

### 2.1 Repository facts checked

| Item | Verified state |
|---|---|
| Git branch | `codex/phase2b-provenance` |
| Git HEAD | `02c079ce6ee1eca19b0d23b750c92a01164a85eb` |
| Frozen task | MAC-only control under exogenous HEART-CH schedules |
| Current main model | `GlobalBranchingDuelingC51` |
| Current model size | 2,842,811 parameters for the recorded Phase 2B seed-2299 configuration |
| Main budget/action | budget 12; per-node actions 0, 1, 2, or 3 slots |
| Queue configuration | maximum 5 packets; TTL 3 rounds |
| Distributional head | C51, 51 atoms, support `[-30, 30]` |
| Frozen Phase 2C reward scale | `0.14436784678738615` |
| Optimizer lineages | 2299, 3299, 4299 |
| Development schedules | 2300-2304 |
| Unrelated pre-existing output | `outputs/phase4/`; preserved and not used here |

### 2.2 Current scientific status

- Phase 0 and Phase 1 environment foundations are complete with their recorded qualifications.
- Phase 2B checkpoints passed their then-registered development checks, but the seed-2299 categorical output audit later found severe upper-support saturation.
- Phase 2C return scaling and provenance wiring are implemented.
- The stronger Phase 2C identity gate failed; consequently the current architecture is rejected as the final policy.
- The earlier five-seed Phase 3 pilot is diagnostic only. It used the older budget-8 policy and cannot validate a replacement model.
- No claim of publication readiness or general superiority is supported.

### 2.3 Literature retrieval boundary

Primary arXiv pages and DOI/publisher landing pages were scraped successfully and archived under:

`F:\WSN\matlab\stage2\hta-mac\.firecrawl\hta_mac_performance_research_20260804`

Firecrawl's research-paper endpoint returned server error 500 for several requested in-body passage extractions. Deep Sets, Action Branching, and IQN passage retrieval succeeded; the other sources were verified through their primary metadata/abstract pages and publisher records. This report does not invent missing in-body quotations or pretend abstract-level evidence is a complete paper audit.

---

## 3. What the current framework does well

The failed identity gate should not obscure the valid engineering already present:

- static and learned policies run through the same environment and energy accounting;
- the cluster schedule is frozen and replayed without stale-frame reuse, isolating the MAC intervention;
- the action projector enforces the hard cluster slot budget;
- queue-feasible caps are applied in both behavior and Bellman targets;
- idle-listening energy is explicit and tested;
- packet generation, FIFO service, stale expiry, overflow, and death drops are explicitly accounted;
- solar and thermal HMM transition-row features are preserved as state-conditioned transition probabilities;
- return scaling has provenance, resume mismatch protection, and categorical-only reinitialization;
- the validation and audit machinery is considerably stronger than the original Phase 2A evaluation.

These pieces should be retained. The repair is targeted at policy representation, policy observability, and objective alignment—not a restart of the simulator.

---

## 4. Measured failure evidence

### 4.1 Branch-identity failure

Artifact:

`outputs/phase2/phase2c_branch_identity_audit_20260804/branch_identity_audit.json`

The audit moved complete state/mask/cap bundles, inverse-mapped allocations, and evaluated all three repaired checkpoints. Results were:

| Optimizer seed | Random-permutation inverse allocation agreement | Raw action agreement | Targeted state-conditioned classification |
|---:|---:|---:|---:|
| 2299 | 0.577443 | 0.953175 | 0.0 |
| 3299 | 0.556241 | 0.923536 | 0.0 |
| 4299 | 0.529674 | 0.920299 | 0.0 |

The registered allocation-agreement gate was at least 0.95. All lineages failed. The high-harvest-more versus high-harvest-fewer targeted mechanism count was 0 versus 0 for every checkpoint.

Interpretation: the projector can sometimes preserve raw local choices, but the allocation as a whole is strongly tied to branch positions. More training cannot establish an architectural symmetry that the model does not possess.

### 4.2 C51 saturation

The seed-2299 Phase 2B checkpoint audit found approximately:

- median Q value: 29.22 on a maximum support value of 30;
- fraction of Q values greater than 29: 63.8%;
- median probability on the top atom: 94.15%.

That makes action differences unreliable and can collapse marginal-gain ranking. The frozen reward scale is a reasonable first repair, but it must be audited again only after a valid architecture learns beyond replay warmup.

### 4.3 Service/freshness trade-off in the old pilot

The old budget-8, five-seed Phase 3 pilot reported approximately:

| Policy | Delivered packets | Fairness | Stale drops |
|---|---:|---:|---:|
| Static | 13,247 | 0.9221 | 17,973 |
| S2A2-style | 22,637 | 0.8403 | 39,389 |
| Old HTA-MAC | 21,374 | 0.7804 | 143,335 |

HTA-MAC was right-censored for FND/HND at the common horizon, but the service and freshness cost was substantial. With five paired observations, Wilcoxon `p = 0.0625`; no significance or superiority claim follows. These numbers are a mechanism warning, not an acceptance comparison for the future model.

### 4.4 Current development baselines

The schema-v2 baseline tuning summary currently selects:

| Policy | Median FND | Delivery ratio | Fairness |
|---|---:|---:|---:|
| Energy proportional | 122 | 0.57025 | 0.95122 |
| Harvest proportional | 130 | 0.54455 | 0.93914 |
| Adapted S2A2 | 204 | 0.36746 | 0.86850 |
| Adapted FFSS | 139 | 0.48637 | 0.97221 |

These are development-tuned reference values. They can inform development feasibility thresholds, but they must not be silently converted into held-out final success gates.

---

## 5. Root-cause analysis

### P0-1. The network is not permutation equivariant

Implementation: `agents/architectures.py`

`GlobalBranchingDuelingC51`:

1. pads to 100 branches;
2. flattens `100 × 50` state features plus a 100-entry mask;
3. processes the resulting ordered vector with a shared MLP;
4. applies one separate advantage head per branch through `nn.ModuleList`.

Both the flattening and node-specific heads encode array position. If physical nodes exchange positions, neither the hidden representation nor the head assignment is guaranteed to exchange accordingly. This exactly predicts the failed audit.

Action Branching was introduced to handle large factored action spaces using a shared decision module and multiple branches. It does not, by itself, guarantee exchangeability of homogeneous entities, variable-cardinality generalization, or invariance to relabelling. The present implementation inherits the branching idea but applies it to homogeneous nodes where permutation symmetry is a required property.

**Decision:** retire this architecture for final evidence. Keep its artifacts for provenance; do not repair it by further training.

### P0-2. Packet expiry is hidden from the policy

Implementation: `envs/intra_cluster_mac_env.py`

The environment maintains exact per-node FIFO lists in `packet_ages`, expires packets beyond TTL, and records stale drops. But `_state()` exposes only normalized queue length, not the age composition, oldest age, or number about to expire.

Two nodes can therefore have identical observed state but different optimal service urgency:

- node A queue: ages `[0, 0, 0]`;
- node B queue: ages `[3, 3, 3]`.

Both appear as queue length 3. Deferring A may be harmless; deferring B causes immediate stale loss. From the policy's view these states alias. The resulting problem is partially observed relative to the freshness dynamics, and no amount of reward tuning can reliably infer unavailable expiry information.

**Decision:** add a normalized TTL age histogram per node before QoS reward experiments. For TTL 3, expose counts for ages 0, 1, 2, and 3 divided by `q_max`; optionally include oldest normalized age and expiring-packet fraction if not algebraically redundant. Add tests that prove distinguishability and transition consistency.

### P0-3. The optimized reward is not the reported service objective

The existing reward contains delivery, idle energy, death, harvest alignment, declining allocation, and queue fairness terms. It has no direct stale-expiry term and no explicit minimum service guarantee. The old pilot's large stale-drop count is therefore not surprising: the policy can trade service away for energy/lifetime benefits without violating a hard requirement.

Timely-throughput/AoI literature formulates service requirements as constraints or virtual queues. That is a better match than repeatedly adjusting an arbitrary stale-drop coefficient.

**Decision:** after the new architecture is validated under the frozen reward, add a development-only constrained objective such as:

- delivery ratio at least `d_min`, and/or
- stale-drop ratio at most `s_max`, and
- fairness at least `j_min`.

Use a Lagrangian multiplier or virtual deficit queue updated from measured violations. Keep the physical slot budget and queue caps hard. Select thresholds from preregistered development evidence and domain requirements, never from final held-out seeds.

### P1-1. Budget projection contains an index tie-break

Implementation: `agents/budget_projection.py`

The heap entries are `(-gain, node, level)`. When marginal gains are equal, Python compares the node index. Thus even a perfectly equivariant Q network can yield label-dependent allocations at exact ties.

Discrete allocation requires some symmetry-breaking rule when identical candidates compete for too few slots. The rule must be explicit and move with the physical-node bundle. Recommended options:

- deterministic per-round priorities derived from a preregistered seed and stable physical-node identity, included in the audit bundle but not as a learned feature; or
- a semantic, measured tie key such as expiring-packet fraction, followed by a seeded stable priority for remaining exact ties.

Do not feed raw node ID into the neural network. Audit Q equivariance separately from projected allocation equivariance, and report tie versus non-tie cases.

### P1-2. Training distribution and evaluation distribution differ

The current dynamic training focuses on a target cluster while the wider network uses static behavior. Final claims concern the network-wide system. Local success can therefore fail during simultaneous policy deployment because energy, queue, and service distributions shift.

**Decision:** use a curriculum:

1. single-cluster mechanism learning;
2. randomized target cluster/rank;
3. mixed learned/static deployment;
4. network-wide learned-policy development gate;
5. only then three-lineage confirmation.

### P1-3. Fixed categorical support remains fragile

C51 projects returns onto a fixed support. When mass accumulates at a boundary, distributional resolution and marginal action ranking degrade. A fixed one-time scale can work only if return geometry remains stable across curricula and constraint penalties.

**Decision:** retain the frozen C51 scale for the first equivariant experiment to preserve attribution. Continuously record lower/upper boundary mass and Q quantiles. If registered occupancy gates fail, run a separate PopArt-normalized C51 experiment. If fixed categorical projection remains the limiting mechanism after that, compare an IQN head in a separate preregistered ablation.

### P2. Validity limitations outside the immediate model repair

- The thermal HMM is a frozen synthetic auxiliary, not a field-trained thermal predictor.
- The CH schedule is exogenous, deliberately preventing joint adaptation.
- The current single-cluster TDMA problem does not explicitly model a link-interference graph.
- Right-censored FND/HND must be reported with Kaplan-Meier/common-horizon restricted event-free time, never as infinity or as the requested horizon.

These limitations should be disclosed. Expanding the scope to solve them now would weaken causal isolation and delay the defensible MAC contribution.

---

## 6. Research literature and direct design implications

### 6.1 Representation and permutation properties

| Primary source | Verified contribution relevant here | HTA-MAC implication |
|---|---|---|
| Zaheer et al., **Deep Sets**, NeurIPS 2017, [arXiv:1703.06114](https://arxiv.org/abs/1703.06114) | Characterizes permutation-invariant set functions and equivariant constructions using shared transforms and commutative aggregation. | Use a shared node encoder and masked invariant aggregation; remove ordered flattening and branch-specific heads. |
| Lee et al., **Set Transformer**, ICML 2019, [arXiv:1810.00825](https://arxiv.org/abs/1810.00825) | Uses attention for interactions among set elements and inducing points for scalable set processing. | Register attention as a later ablation if mean/max pooling lacks relational capacity; it is not required for the first repair. |
| Zhao, Yang, and Liu, **GNNs with Desired Permutation Properties for Wireless Networks**, [arXiv:2203.03906](https://arxiv.org/abs/2203.03906) | Derives processing, combining, and pooling choices that preserve desired permutation properties in heterogeneous wireless policies. | Make symmetry an architectural invariant and test it numerically; do not rely on data augmentation alone. |
| Wu, Sun, and Yang, **Size Generalization of GNNs for Wireless Link Scheduling**, [arXiv:2204.13972](https://arxiv.org/abs/2204.13972) | Studies how aggregation and activation affect generalization across network sizes, including mean aggregation. | Prefer masked mean over unnormalized sum for variable active-node counts; include active fraction explicitly. |
| Zhao et al., **GNN-based Link Scheduling**, IEEE TWC, [arXiv:2109.05536](https://arxiv.org/abs/2109.05536), [DOI](https://doi.org/10.1109/TWC.2022.3222781) | Combines learned graph embeddings with a scheduling/selection mechanism and studies generalization. | Learned equivariant scores plus a deterministic hard-budget projector is a sound decomposition; a graph network is optional, not mandatory here. |

### 6.2 Large discrete action spaces and constraints

| Primary source | Verified contribution relevant here | HTA-MAC implication |
|---|---|---|
| Tavakoli et al., **Action Branching Architectures for Deep Reinforcement Learning**, AAAI 2018, [arXiv:1711.08946](https://arxiv.org/abs/1711.08946) | Uses a shared decision module with multiple action branches to reduce combinatorial action output growth. | Retain factored per-node actions, but replace node-position-specific branches with a shared branch function over each node embedding. |
| Bhatia et al., **Constrained Combinatorial Optimization with RL**, [arXiv:2006.11984](https://arxiv.org/abs/2006.11984) | Treats constrained combinatorial decisions through constrained RL and penalty signals. | Keep feasibility in the projector and represent QoS violations explicitly rather than hoping an unconstrained reward discovers acceptable service. |
| Fountoulakis et al., **AoI Minimization with Timely-Throughput Constraints**, accepted IEEE Transactions on Communications, [arXiv:2109.04784](https://arxiv.org/abs/2109.04784) | Formulates freshness scheduling with timely-throughput constraints and Lyapunov/virtual-queue mechanisms. | Treat delivery/freshness as service constraints; packet age must be observable to the policy. |

### 6.3 Distributional stability

| Primary source | Verified contribution relevant here | HTA-MAC implication |
|---|---|---|
| Bellemare, Dabney, and Munos, **A Distributional Perspective on RL**, ICML 2017, [arXiv:1707.06887](https://arxiv.org/abs/1707.06887) | Introduces C51's categorical return distribution and fixed-support projection. | Boundary occupancy is a first-class correctness metric, not only a diagnostic plot. |
| Rowland et al., **An Analysis of Categorical Distributional RL**, AISTATS 2018, [arXiv:1802.08163](https://arxiv.org/abs/1802.08163) | Analyses categorical projection and approximation behavior. | Keep support/projection effects separate from representation failure in attribution experiments. |
| van Hasselt et al., **Learning Values Across Many Orders of Magnitude (PopArt)**, NeurIPS 2016, [arXiv:1602.07714](https://arxiv.org/abs/1602.07714) | Adaptively normalizes targets while preserving unnormalized outputs. | First escalation if a fixed reward scale fails across curricula or constraint multipliers. |
| Dabney et al., **Implicit Quantile Networks**, ICML 2018, [arXiv:1806.06923](https://arxiv.org/abs/1806.06923) | Represents the return quantile function without fixed categorical atoms. | A later head-level alternative if C51 remains boundary-limited after controlled normalization experiments. |

### 6.4 EH scheduling and closest MAC context

| Primary source | Verified contribution relevant here | HTA-MAC implication |
|---|---|---|
| Sharma et al., **Structure-Aware RL for Energy Harvesting Devices**, [arXiv:1807.08315](https://arxiv.org/abs/1807.08315) | Exploits monotonicity/increasing-difference structure involving queue and battery state. | After observability and symmetry are fixed, test a monotonicity regularizer or structural gate as a sample-efficiency ablation. |
| HENO-MAC, IEEE WCNC 2024, [arXiv:2401.00717](https://arxiv.org/abs/2401.00717), [DOI](https://doi.org/10.1109/WCNC57260.2024.10571258) | Energy-neutral MAC for hybrid solar-wind harvesting with delay evaluation and realistic energy traces. | Use as a close hybrid-energy MAC comparator in positioning; it does not validate the present solar/thermal state representation or slot-allocation mechanism. |
| Dutta et al., cooperative EH RL, IEEE TGCN 2025, [DOI](https://doi.org/10.1109/TGCN.2025.3544073) | Close recent cooperative EH/RL work. | Preserve precise differentiation; do not claim first EH RL or first harvest-aware scheduling. |
| Dutta et al., contextual slot/sleep DRL, IEEE TGCN 2024, [DOI](https://doi.org/10.1109/TGCN.2024.3358230) | Close slot/sleep control work. | Compare action granularity, state, hard budget, cluster-schedule isolation, and evaluation—not broad novelty labels. |
| S2A2MAC, International Journal of Communication Systems, [DOI](https://doi.org/10.1002/dac.5202) | Adaptive EH MAC baseline in the existing competitor set. | Keep the adapted baseline implementation explicit and avoid implying source-code equivalence. |
| FFSS/AFSS, IET Communications, [DOI](https://doi.org/10.1049/cmu2.12243) | Frame/slot scheduling comparator used in the present baseline map. | Retain as a rule-based service/energy reference with documented adaptation. |
| SHR-TDMA, IET Communications, [DOI](https://doi.org/10.1049/iet-com.2019.0977) | Harvest-aware TDMA comparator. | Prevent unsupported “first harvest-aware TDMA” claims. |

### 6.5 What not to infer from the literature

- Deep Sets does not prove that the proposed HTA-MAC network will improve lifetime or delivery; it supplies the needed symmetry construction.
- Wireless GNN results do not justify adding routing or an interference graph absent from this environment.
- Action Branching does not validate node-specific heads for exchangeable nodes.
- PopArt or IQN does not repair hidden state or branch identity.
- HENO-MAC's reported delay improvements are not transferable effect sizes for this simulator.

---

## 7. Recommended model: EquivariantSetBranchingC51

### 7.1 Input representation

For each active node `i`, construct a feature vector `x_i` containing the existing physical/HMM features plus:

- age histogram `age_i[0:TTL] / q_max`;
- current feasible action cap encoded as either normalized scalar plus an action-validity mask, or a four-entry validity vector;
- alive/active mask where needed;
- no learned raw node-ID feature.

Keep the state generator semantic and testable. If the learned embedding expands 18 physical features to 50, document precisely whether the age/cap block is appended before or after that embedding and update schema hashes.

### 7.2 Network equations

For the active-node mask `m_i`:

```text
h_i       = phi(x_i)                                  shared node encoder
g_mean    = masked_mean_i(h_i)
g_max     = masked_max_i(h_i)
g         = rho([g_mean, g_max, active_fraction,
                 normalized_budget, cluster/global features])
V         = value(g)                                  atoms
A_i(a)    = advantage([h_i, g, valid_action_mask_i]) shared local head
Z_i(a)    = V + A_i(a) - mean_valid_action A_i(a)
```

The same `phi` and advantage network must process every node. Aggregation must ignore padding. Invalid actions must be masked before behavior selection, target selection, and projection.

The expected property for a node permutation `P` is:

```text
Q(PX, PM, PC) = P Q(X, M, C)
```

up to registered floating-point tolerance. The parameter count must be independent of the configured maximum number of nodes, apart from non-learned buffers.

### 7.3 Why mean plus max first

- Mean is stable when active cluster size changes and prevents representation magnitude from growing solely with node count.
- Max exposes an urgent/extreme node signal that a mean can dilute, such as a nearly expired backlog.
- Active fraction and normalized budget preserve cardinality information lost by a mean.
- This construction is simpler to audit than attention and matches the present contention-free, cluster-local problem.

### 7.4 When to consider attention or a GNN

Run a Set Transformer ablation only if the pooled model passes symmetry but cannot learn the registered state-conditioned action mechanism or network-wide service gate. Add a GNN only if a real pairwise relation—interference, topology, or explicit link compatibility—is introduced within the approved MAC scope. Do not add a graph merely because wireless papers use GNNs.

### 7.5 Projector contract

The network outputs marginal action values; the projector remains responsible for:

- `sum_i a_i <= budget`;
- `0 <= a_i <= cap_i`;
- deterministic, documented tie handling;
- inverse-equivariant behavior when the full node bundle, including the tie-priority token, is permuted.

Record separately:

1. Q/logit equivariance error;
2. local argmax agreement;
3. projected allocation agreement outside ties;
4. projected allocation agreement including exact ties;
5. budget and cap feasibility.

---

## 8. Controlled improvement sequence

### Experiment A: representation and observability repair

Change only:

- age/expiry features;
- equivariant set-branching architecture;
- semantic tie handling.

Hold fixed:

- simulator physics and accounting;
- frozen CH schedules;
- reward weights and Phase 2C scale;
- C51 support and atom count;
- budget, queue, and TTL configuration;
- development schedule set.

Train from scratch. Old checkpoints are structurally incompatible and identity-tainted; they must not initialize the new model.

### Experiment B: learning mechanics and distributional health

Only after Experiment A's untrained symmetry tests pass:

- run seed 2299 beyond replay warmup;
- cover at least two complete 25-pair development cycles;
- audit gradients, parameter change, replay insertion/sampling, target synchronization, action diversity, and categorical boundary mass;
- stop immediately on a registered gate failure.

### Experiment C: QoS constraint

Only after the architecture learns a valid state-conditioned mechanism:

- define development-only delivery/stale/fairness thresholds;
- implement a Lagrangian or virtual-queue constraint signal;
- compare frozen-reward versus constrained objective using the same seeds and schedules;
- report the complete lifetime/service Pareto trade-off, not one selected metric.

### Experiment D: distributional escalation

Run only if C51 boundary gates fail:

1. PopArt-normalized C51 while preserving unnormalized value semantics;
2. if still necessary, a separate IQN-head ablation.

Do not change the set encoder, constraint mechanism, and distributional head in the same comparison.

### Experiment E: structural regularization

After the base model succeeds, test as an ablation—not a hidden default—a monotonicity/ordering penalty for carefully controlled state pairs:

- higher expiring backlog should not reduce service priority, all else equal;
- higher available harvest may support greater allocation, all else equal;
- critically low energy may legitimately counteract those directions, so pair construction must hold confounders fixed and use qualified expectations rather than universal monotonic laws.

---

## 9. Layer-wise execution plan

### Phase 2D-0 — Freeze and provenance

**Dependencies:** current HEAD and existing Phase 2C failure artifact  
**Owner:** research/provenance layer

- [ ] Hash the identity audit, current checkpoints, configuration, and this report.
- [ ] Register the new architecture name and state-schema version.
- [ ] Scan every prior artifact before selecting untouched final-evaluation seeds.
- [ ] Mark seeds 3100-3104 as already touched by the earlier pilot.
- [ ] Record that `outputs/phase4/` is unrelated pre-existing work and leave it unchanged.

**Exit criteria:** a machine-readable manifest records paths, hashes, branch, commit, scope, dev seeds, touched seeds, and frozen parameters.

### Phase 2D-1 — State sufficiency

**Dependencies:** 2D-0  
**Owner:** environment/state layer  
**Primary files:** `envs/intra_cluster_mac_env.py`, state/embedding configuration, validation tests

- [ ] Add normalized packet-age bins for ages 0 through TTL.
- [ ] Add/verify action-cap encoding and validity mask.
- [ ] Version the physical-state and learned-state schemas.
- [ ] Test two equal-length queues with different ages produce different states.
- [ ] Test FIFO service decrements the correct bins.
- [ ] Test aging, expiry, overflow, and death drops match the emitted state.
- [ ] Test all features are finite and scale-bounded.
- [ ] Confirm static baselines and accounting are unchanged.

**Exit criteria:** state transition tests pass; queue urgency is observable; no energy/accounting regression.

### Phase 2D-2 — Equivariant architecture and projector

**Dependencies:** 2D-1  
**Owner:** model/action layer  
**Primary files:** `agents/architectures.py`, `agents/budget_projection.py`, model factory/config, validation tests

- [ ] Implement `EquivariantSetBranchingC51` with shared `phi`, masked mean/max, global value, and shared advantage head.
- [ ] Mask padding and invalid actions in all behavior and target paths.
- [ ] Remove index-based tie resolution; implement bundle-stable deterministic priority.
- [ ] Add model serialization metadata and architecture mismatch rejection.
- [ ] Measure parameter count and peak memory; do not estimate them in the paper.
- [ ] Test multiple active-node counts and padding layouts.

Registered untrained and randomly initialized gates:

- [ ] maximum absolute Q/logit inverse-equivariance error `<= 1e-6` in float32;
- [ ] local argmax inverse agreement `= 1.0` outside numerical ties;
- [ ] projected allocation agreement `>= 0.99` outside exact ties;
- [ ] projected allocation agreement `>= 0.95` overall using moved tie-priority bundles;
- [ ] 20 deterministic random permutations plus at least 10 targeted swaps per initialization;
- [ ] 100% budget and cap feasibility.

**Exit criteria:** every gate passes before any costly training.

### Phase 2D-3 — Learning-mechanics smoke

**Dependencies:** 2D-2  
**Owner:** RL training layer

- [ ] Fresh seed-2299 initialization; do not load Phase 2B/2C weights.
- [ ] Run long enough to cross replay warmup and complete at least 50 episodes/two 25-pair cycles.
- [ ] Verify nonzero finite gradients, parameter updates, replay sampling, and target-network updates.
- [ ] Archive action histograms, cap/budget utilization, state-stratified marginal Q, and reward components.
- [ ] Re-run identity audit on the trained checkpoint.
- [ ] Audit lower/top atom probability, Q quantiles, return quantiles, and clipping/projection counts.
- [ ] Require state-conditioned high-harvest/declining and expiry-urgency tests to show a learned difference in the registered direction on controlled pairs.

**Stop conditions:** identity regression, nonfinite values, replay failure, absent parameter change, invalid allocation, or categorical boundary saturation above the preregistered limit.

**Exit criteria:** mechanics and identity pass with archived printed evidence; no claim of policy superiority yet.

### Phase 2D-4 — Development attribution and QoS

**Dependencies:** 2D-3  
**Owner:** objective/evaluation layer

- [ ] Run representation-only and age-feature ablations to separate their effects.
- [ ] Define `d_min`, `s_max`, and `j_min` from development evidence/domain requirements before evaluating candidates.
- [ ] Implement and test constraint multiplier or virtual-queue updates.
- [ ] Report multiplier trajectories and violation frequency; prevent reward-scale drift from going unseen.
- [ ] Compare lifetime, idle energy, delivery, stale/overflow/death drops, fairness, action diversity, and constraint violations.
- [ ] If fixed C51 fails boundary gates, execute PopArt-C51 as a separate registered experiment.
- [ ] Do not access final held-out seeds.

**Exit criteria:** one configuration selected by a predeclared multi-metric development rule, with complete rejected-candidate records.

### Phase 2D-5 — Network-wide development gate

**Dependencies:** 2D-4  
**Owner:** system-integration layer

- [ ] Progress through target-cluster, randomized cluster, mixed deployment, and all-cluster deployment.
- [ ] Verify schedule alignment and no stale-frame replay at every stage.
- [ ] Compare against tuned static, energy-proportional, harvest-proportional, adapted S2A2, and adapted FFSS baselines.
- [ ] Use common horizons and censor-aware lifetime summaries.
- [ ] Require service constraints and identity gates to remain satisfied network-wide.

**Exit criteria:** policy meets the preregistered development Pareto/constraint rule under all-cluster deployment.

### Phase 2D-6 — Three-lineage confirmation

**Dependencies:** 2D-5  
**Owner:** reproducibility layer

- [ ] Train fresh lineages 2299, 3299, and 4299 with identical frozen protocol.
- [ ] Re-run identity, mechanism, categorical, and network-wide gates per lineage.
- [ ] Report dispersion and failures; do not average away a failed mandatory gate.
- [ ] Freeze one publication candidate only if all mandatory gates pass.

**Exit criteria:** a hashed candidate checkpoint/config plus complete three-lineage evidence.

### Phase 3R — Untouched paired evaluation

**Dependencies:** 2D-6 and frozen unused seed registry  
**Owner:** statistical evaluation layer

- [ ] Select and freeze 30 unused schedule seeds only after the prior-seed scan.
- [ ] Run paired common-schedule comparisons for every policy.
- [ ] Report FND/HND events, censoring, Kaplan-Meier curves, and common-horizon restricted event-free time.
- [ ] Report delivery ratio, timely delivery/stale ratio, fairness, energy, and constraint satisfaction with confidence intervals.
- [ ] Use paired Wilcoxon tests with multiplicity control and effect sizes where appropriate.
- [ ] Publish failures and negative trade-offs; do not replace censored lifetimes with infinity or the horizon.

**Exit criteria:** locked, reproducible statistical artifact; only then decide whether the evidence supports a manuscript claim.

### Phases 4R-6R — Ablations, figures, manuscript

- [ ] Ablate age histogram, HMM transition block, thermal source, mean/max context, constraint mechanism, and distributional normalization one factor at a time.
- [ ] Validate any empirical lifetime bound using the same accounting and censoring rules.
- [ ] Update the competitor audit with primary sources and exact differentiation.
- [ ] Build a claim-to-code-to-artifact traceability table.
- [ ] State that thermal prediction is synthetic unless new validated data are added in a separately approved scope.

---

## 10. Recommended metrics and mandatory plots

### Identity and mechanism

- maximum/median absolute Q equivariance error;
- raw action and projected allocation inverse agreement;
- tie prevalence and tie-conditioned agreement;
- marginal-Q curves by energy, forecast, queue age, and cap;
- controlled pair response for harvest, energy decline, queue urgency, and expiry.

### Learning health

- replay size and sampled-age distribution;
- gradient and parameter-update norms;
- target-network lag/update count;
- action entropy and per-action frequency;
- slot-budget utilization and unused-positive-gain cases;
- C51 top/bottom atom mass and Q/return quantiles.

### System outcomes

- FND/HND event indicators and censoring times;
- Kaplan-Meier and restricted event-free time;
- packets generated, delivered, stale-dropped, overflow-dropped, and death-dropped;
- timely delivery ratio and total delivery ratio;
- Jain fairness and per-node service distribution;
- idle, transmit, receive, and total energy where accounting supports them;
- residual-energy distribution, not only its mean;
- constraint violations and multiplier/virtual-queue trajectories.

Every plot must identify commit, config hash, checkpoint hash, schedule seeds, optimizer seed, horizon, and whether the result is development or untouched evaluation.

---

## 11. Approaches rejected or deferred

| Approach | Decision | Reason |
|---|---|---|
| Continue training current node-specific architecture | Reject | Cannot learn guaranteed permutation symmetry; identity gate already failed. |
| Permutation augmentation alone | Reject as repair | May reduce empirical sensitivity but does not remove ordered flattening or indexed heads. |
| Feed node ID to the model | Reject | Encodes the nuisance identity the audit is intended to remove. |
| Add a GNN immediately | Defer | No explicit interference/topology relation is part of the cluster-local action problem; a set model is the minimal valid representation. |
| Add attention immediately | Defer to ablation | More complex and harder to audit; mean/max pooled context may be sufficient. |
| Switch directly from C51 to IQN | Defer | Would confound representation, observability, and return-distribution repairs. |
| Retune all reward weights with the new network | Reject initially | Destroys causal attribution; first keep the frozen reward and scale. |
| Penalize stale drops without exposing age | Reject | The policy still cannot distinguish imminent expiry states. |
| Warm-start from Phase 2B/2C | Reject | Architecture is incompatible and learned values are identity-tainted/saturated. |
| Jointly learn CH selection or routing | Out of scope | Violates frozen HEART-CH causal isolation. |
| Use HERMES outputs | Reject | Simulator assumptions are incompatible/disconnected. |

---

## 12. Expected performance effects and honest uncertainty

The following are hypotheses, not promised improvements:

- **Equivariant sharing** should improve data efficiency and robustness by making every node update the same encoder/head and eliminating position memorization.
- **Age observability** should reduce stale drops because service urgency becomes distinguishable, but it may reduce apparent lifetime if the policy spends more energy on timely service.
- **Explicit service constraints** should bound that trade-off, but feasibility may be impossible at aggressive thresholds under budget 12; infeasibility must be reported rather than hidden by multiplier growth.
- **Mean/max aggregation** may generalize better across active-node counts; however, it may lose pairwise detail. Attention is the registered escalation if controlled evidence shows this limitation.
- **PopArt** may stabilize value targets as penalties/curricula change, but cannot compensate for a defective policy state.
- **Structural monotonicity losses** may improve sample efficiency, but incorrect universal monotonic assumptions could bias the policy under low energy or queue-cap interactions.

No numerical gain should be stated until the corresponding artifact exists. The key next success criterion is first **correctness of mechanism**, then **development feasibility**, then **untouched performance**.

---

## 13. Immediate implementation order

The shortest defensible path from the current repository is:

1. freeze the present failure evidence and state schema;
2. add TTL age histogram and cap/action mask features with transition tests;
3. implement `EquivariantSetBranchingC51` and bundle-stable tie handling;
4. pass untrained permutation/property tests;
5. run a fresh seed-2299 smoke through replay warmup and two development cycles;
6. audit learned identity, state-conditioned behavior, and C51 boundary mass;
7. only after those pass, add and compare an explicit QoS constraint;
8. progress to network-wide development and three-lineage confirmation;
9. freeze unused seeds and conduct a new 30-seed paired, censor-aware evaluation.

The current Phase 2C task sequence should therefore be superseded by Phase 2D. Running the planned old-architecture learning smoke would consume compute without resolving the already-failed mandatory identity property.

---

## 14. Reproducibility commands

Run from `F:\WSN\matlab\stage2\hta-mac`:

```powershell
git branch --show-current
git rev-parse HEAD
git status --short
python -B -m pytest validation -q -p no:cacheprovider
```

Before any final evaluation, archive:

```text
commit SHA
dirty/untracked status
configuration and schema hashes
checkpoint SHA-256
optimizer and schedule seeds
software/package versions
exact command line
stdout/stderr log
machine-readable metrics
```

Do not delete or overwrite old artifacts. Use a new output directory for each gate and fail if the directory already contains a different manifest.

---

## 15. Firecrawl source collection log

### Search/research themes

- permutation-equivariant set policies and wireless GNNs;
- large discrete/factored action spaces;
- constrained RL for combinatorial scheduling;
- categorical distributional support, PopArt, and quantile alternatives;
- energy-harvesting scheduling, timely throughput, AoI, and hybrid-energy MAC.

### Locally archived primary pages

The evidence directory includes primary-page snapshots for arXiv IDs:

`1602.07714`, `1703.06114`, `1707.06887`, `1711.08946`, `1802.08163`, `1806.06923`, `1807.08315`, `1810.00825`, `2006.11984`, `2109.04784`, `2109.05536`, `2203.03906`, `2204.13972`, and `2401.00717`.

It also includes DOI landing-page snapshots for:

`10.1109/TGCN.2025.3544073`, `10.1109/TGCN.2024.3358230`, `10.1002/dac.5202`, `10.1049/cmu2.12243`, and `10.1049/iet-com.2019.0977`.

### Rerun inputs

Use the following conceptual inputs when refreshing the search; do not paste API credentials into scripts or reports:

```text
equivariant set neural network homogeneous wireless scheduling
permutation equivariant GNN wireless resource allocation size generalization
factored multidimensional discrete actions action branching reinforcement learning
constrained combinatorial reinforcement learning timely throughput AoI
categorical distributional reinforcement learning support saturation PopArt IQN
energy harvesting MAC hybrid energy TDMA reinforcement learning queue age
```

Refresh the literature audit before submission because the search date is 4 August 2026 and new work may appear.

---

## 16. Final recommendation

Proceed with **Phase 2D: state-sufficient permutation-equivariant set branching**. Do not spend additional optimization budget on the current flattened/node-head architecture. The next model must first prove that relabelling nodes relabels its outputs, that packet expiry urgency is observable, and that its hard-budget projector does not reintroduce label bias. Only then should reward constraints, normalization, or more expressive set interaction be evaluated.

This path preserves the strongest parts of the existing HTA-MAC work, directly repairs the measured failures, and creates a sequence in which every claimed improvement has a controlled attribution and a falsifiable exit gate.
