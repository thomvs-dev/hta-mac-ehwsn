# HTA-MAC Publication Rescue: Research, Mathematics, and Fast Execution Plan

**Date:** 3 August 2026  
**Purpose:** Convert the completed but weakly trajectory-responsive Phase 2
models into a defensible journal paper without fabricating results or restarting
the whole research programme.  
**Scope remains fixed:** intra-cluster MAC only; frozen exogenous HEART-CH
schedule; no routing and no CH retraining.

## 1. Bottom-line recommendation

The paper can still become publishable, but not by submitting the present
checkpoints or merely increasing a reward coefficient. The fastest technically
defensible rescue is to change the algorithm from an unconstrained Branching
DQN to a **structure-constrained, counterfactual trajectory-regularized
Branching DQN**, warm-start it from the completed models, and fine-tune it.

The revised paper should be built around four connected ideas:

1. physically normalized hybrid-harvest inputs so the network can actually
   perceive them;
2. counterfactual structure regularization that directly trains the marginal
   slot values to respond to HMM trajectory changes;
3. diminishing-return branch utilities, which make the greedy budget
   projection an exact optimizer of the learned separable surrogate;
4. explicit idle-energy accounting and a paired lifetime/QoS frontier under a
   fixed CH schedule.

This is not an artificial patch. Structure-aware RL is an established research
direction. Sharma, Mastronarde, and Chakareski derive value-function structure
for EH sensor scheduling and exploit it to accelerate RL
([IEEE TSP 2020](https://doi.org/10.1109/TSP.2020.2973125)). Chen et al. insert
proved scheduling structure into action selection and the DRL loss and report
substantial reductions in training time
([IEEE TWC 2024](https://doi.org/10.1109/TWC.2023.3277861)). Constrained
monotonic neural networks provide a modern neural basis for imposing domain
monotonicity without losing universal approximation over continuous monotone
functions ([ICML 2023](https://proceedings.mlr.press/v202/runje23a.html)).

The existing 18 checkpoints remain valuable as:

- warm-start initializations;
- the unregularized BDQ ablation;
- evidence motivating the new structure-aware method;
- a transparent negative result showing why raw Q sensitivity is insufficient.

No publication venue can be guaranteed. The plan below gives the paper a
coherent algorithmic contribution and the evidence a competent reviewer is
likely to demand.

## 2. What the completed training revealed

The August 3 Colab artifact is valid and stable, but the intended mechanism is
weak:

- 18/18 registered runs passed the implemented numerical gate;
- no run collapsed to all-sleep;
- original S1/S8 checks changed the local greedy action in 0/18 runs;
- the expanded audit changed the probed node's projected slot count in only
  3/1,764 counterfactuals;
- S8 received more slots in 0/1,764 counterfactuals.

The important new local measurement is input scaling. Across initial active
development nodes:

| Input block | Observed scale |
|---|---:|
| Normalized residual energy | 1 at reset; generally order 1 |
| Hybrid forecast mean | `6.484e-5` to `3.749e-4` |
| Hybrid forecast variance | `1.549e-8` to `4.239e-8` |
| HMM transition probabilities | approximately 0 to 0.85 |
| Frozen ST-GCN embedding | 0 to 47.075; standard deviation 7.169 |

The current shared network applies one LayerNorm to the flattened 5,100-value
global input. That operation does not normalize each physical feature. The
forecast mean is roughly four orders of magnitude below the embedding scale,
and forecast variance is roughly eight to nine orders below it. The first
linear layer has no compensating large weights on those features; its mean
absolute weights are actually smaller for forecast mean/variance than for most
other blocks.

This makes feature-scale suppression the first actionable diagnosis. It does
not completely explain the weak transition-vector response, because transition
probabilities are order one; therefore normalization alone is necessary but not
sufficient.

## 3. Literature-backed scientific direction

### 3.1 Why a structure-aware rescue is credible

The relevant methodological line is not “try another generic RL algorithm.” It
is to exploit known scheduling structure:

- Structure-aware EH scheduling has already been shown to improve learning
  efficiency by using monotone/increasing-difference properties of battery and
  queue states ([Sharma et al., IEEE TSP 2020](https://doi.org/10.1109/TSP.2020.2973125)).
- Structure-enhanced DRL has been implemented through both action-selection
  guidance and an auxiliary structural loss
  ([Chen et al., IEEE TWC 2024](https://doi.org/10.1109/TWC.2023.3277861)).
- Smooth policy regularization can reduce the effective search space and
  improve sample efficiency and robustness
  ([Shen et al., ICML 2020](https://proceedings.mlr.press/v119/shen20b.html)).
- Constrained monotone architectures can encode prior monotonic relationships
  by construction ([Runje and Shankaranarayana, ICML 2023](https://proceedings.mlr.press/v202/runje23a.html)).
- BDQ remains appropriate for avoiding combinatorial joint-action output; its
  shared decision module and linear branch growth are the original
  architectural motivation
  ([Tavakoli et al., AAAI 2018](https://doi.org/10.1609/aaai.v32i1.11798)).

### 3.2 Why the paper must remain narrowly differentiated

The expanded literature search rules out broad novelty claims:

- SHR-TDMA already uses hybrid EH characterization and per-node TDMA
  assignment ([Gong et al., 2020](https://doi.org/10.1049/iet-com.2019.0977)).
- Ge, Nan, and Guo already use per-node cooperative Q-learning/SARSA with
  predicted next-slot solar harvest in a clustered EH-WSN
  ([2021](https://doi.org/10.1177/15501477211007411)).
- FFSS/AFSS already perform forecast-aware EH slot scheduling
  ([2021](https://doi.org/10.1049/cmu2.12243)).
- Seifullaev et al. combine first-order Markov harvesting, Bayesian estimation,
  and learned transmission policies
  ([IEEE TGCN 2024](https://doi.org/10.1109/TGCN.2024.3374899)).
- Dutta et al. already study contextual DRL for slot allocation and
  transmit-sleep control
  ([IEEE TGCN 2024](https://doi.org/10.1109/TGCN.2024.3358230)).

The defensible distinction is the conjunction of clustered terrestrial
hybrid-HMM trajectory features, a shared branching multi-slot policy, a hard
cluster budget, explicit idle accounting, and a fixed exogenous CH schedule.
The structure-aware revision makes that conjunction technically meaningful
rather than merely listing unused inputs.

## 4. Revised mathematical formulation

For active member node (i), let the feasible slot count be

\[
a_i\in\{0,1,\ldots,c_i\},\qquad
c_i=\min(q_i,n_{\max}),\qquad
\sum_i a_i\le T.
\]

The C51 branch gives expected action values

\[
Q_i(k\mid s)=\sum_{j=1}^{N_{\mathrm{atom}}}z_j
p_{i,k,j}(s),
\]

and the value of the (k)-th additional slot is the marginal gain

\[
d_{i,k}(s)=Q_i(k\mid s)-Q_i(k-1\mid s),
\quad k=1,\ldots,n_{\max}.
\]

This marginal value—not raw Q distance—is what the budget projector uses and
what the scientific diagnostic must test.

### 4.1 Physically normalized state

Using HEART-CH's rectified moments, define

\[
H_{\max}=\max_m g_{1,s}(m)+\max_n g_{1,h}(n).
\]

For the current frozen HMM parameters,

\[
H_{\max}=3.74939\times10^{-4}\ \mathrm{J/round}.
\]

Use

\[
\bar\mu_i=\widehat\mu_i/H_{\max},\qquad
\bar v_i=\widehat v_i/H_{\max}^2.
\]

The ST-GCN embedding should be normalized per node,

\[
\bar z_i=\operatorname{LayerNorm}(z_i),
\]

before concatenation. Energy, queue, previous slots, and cluster fraction are
already normalized. HMM transition vectors remain probability simplexes.

This is deterministic physical normalization; it introduces no result-tuned
constant.

### 4.2 Diminishing-return constraint

Multiple slots for one queued node should not have increasing learned marginal
utility without evidence. Impose discrete concavity:

\[
d_{i,1}\ge d_{i,2}\ge\cdots\ge d_{i,n_{\max}}.
\]

During training use

\[
\mathcal L_{\mathrm{conc}}=
\frac{1}{B}\sum_{b,i}\sum_{k=1}^{n_{\max}-1}
\left[\max\{0,d_{i,k+1}^{(b)}-d_{i,k}^{(b)}\}\right]^2.
\]

At inference, apply a non-increasing isotonic projection to each marginal
vector. This hard step guarantees concavity even if the soft training penalty
has small residual violations.

**Proposition 1—exact projection.** If each branch utility is discretely
concave, repeatedly assigning the next slot with the largest positive feasible
marginal gain solves

\[
\max_a\sum_i Q_i(a_i\mid s)
\quad\text{subject to}\quad
0\le a_i\le c_i,\ \sum_i a_i\le T
\]

exactly for the projected separable surrogate. The proof is an exchange
argument: any allocation not containing a larger available prefix marginal can
exchange a selected smaller marginal for it without violating prefix
feasibility. Repeating the exchanges yields the greedy allocation. This is the
standard separable-concave integer resource-allocation setting; formal greedy
optimality conditions are treated by
[Shenmaier (2003)](https://doi.org/10.1016/S0166-218X(03)00435-9).

This replaces the current vague “knapsack-style” analogy with a precise
condition and guarantee.

### 4.3 Counterfactual trajectory-order loss

For a replay state, form a matched pair for node (i):

- (s_i^+): higher risk-adjusted hybrid forecast;
- (s_i^-): lower risk-adjusted hybrid forecast;
- identical residual energy, queue, previous slots, cluster membership,
  embedding, and every other node.

The alternatives must come from actual HMM rows and rectified moments, not
arbitrary Gaussian noise. Define

\[
\rho_i=\frac{\widehat\mu_i-\kappa\sqrt{\widehat v_i}}{H_{\max}}.
\]

Use (kappa=0) initially to enforce expected-harvest order without inventing
risk aversion. Treat (kappa\in\{0,0.5,1\}) only as a declared development
ablation if time permits.

For pairs with (ho_i^+>\rho_i^-), train

\[
\mathcal L_{\mathrm{traj}}=
\frac{1}{B n_{\max}}
\sum_{b,i,k}
\max\{0,m+d_{i,k}(s_i^-)-d_{i,k}(s_i^+)\}.
\]

The margin should be scale-relative,

\[
m=\eta\;\operatorname{stopgrad}
\left(\operatorname{median}_{b,i,k}|d_{i,k}|\right),
\]

with (eta\) selected only on development seeds from a small frozen set such
as ({0.02,0.05,0.10}). This prevents a meaningless (10^{-4}) pass while
avoiding a unit-dependent magic number.

The loss encodes a conditional statement, not “always allocate to high
harvest”: when energy, queue, topology, and identity are equal, a higher
available-energy trajectory should not have a lower learned marginal service
value. Other state differences remain free to override it in normal operation.

### 4.4 Auxiliary harvest prediction

Add a small head from the shared representation to the realized normalized
next-round harvest:

\[
\widehat h_{i,t+1}=g_\psi(u_i),\qquad
\mathcal L_H=\operatorname{Huber}
\left(\widehat h_{i,t+1},H_{i,t+1}/H_{\max}\right).
\]

This forces the learned representation to retain harvest-dynamics information.
It does not by itself prove policy use, which is why the counterfactual marginal
loss remains necessary. Return-aware auxiliary representation losses have
precedent in RL representation learning
([ICLR 2021](https://openreview.net/forum?id=VUSDbZRYWFp)).

### 4.5 Complete fine-tuning objective

\[
\boxed{
\mathcal L=
\mathcal L_{\mathrm{C51}}
+\lambda_t\mathcal L_{\mathrm{traj}}
+\lambda_c\mathcal L_{\mathrm{conc}}
+\lambda_h\mathcal L_H
}
\]

Select ((\lambda_t,\lambda_c,\lambda_h,\eta)) only on development seeds and
freeze them before the new paired test. Do not choose the combination from
lifetime results alone. Use the following lexicographic development rule:

1. no collapse or non-finite values;
2. trajectory-use gate passes;
3. concavity violation after training is below 1% before hard projection;
4. among passing configurations, maximize development reward;
5. break ties by lower runtime.

## 5. Fast implementation path that reuses the current models

### Stage A—minimal repair, no architecture restart

1. Add the deterministic physical input normalizer.
2. Add replay-time HMM counterfactual pairs.
3. Add trajectory and concavity losses.
4. Add the small next-harvest prediction head.
5. Load each August 3 checkpoint and fine-tune for 100-150 episodes with a low
   learning rate, initially `2e-5` or `1e-5`.
6. Retain optimizer-independent target networks and replay warm-up; do not load
   stale optimizer momentum unless verified compatible with the new loss.

This is the preferred rescue because the existing networks already learned the
queue/energy/lifetime trade-off. Fine-tuning needs to teach feature scaling and
trajectory structure, not relearn the simulator.

### Stage B—only if Stage A still fails

Add a shared local trajectory residual head to each node branch. It receives
only the normalized 14-value harvest block and contributes to the branch
action logits before the C51 atom softmax. Zero-initialize its final layer so a
loaded checkpoint initially reproduces the old policy. Fine-tune the residual
head and last shared layer first, then unfreeze the full network.

This gives the trajectory block a direct path to its own node's action branch
and prevents the 5,100-dimensional global flattening from being its only path.
It is still a shared branching architecture; parameter growth remains linear
in active branches and the residual head weights are shared across nodes.

Do not jump immediately to PPO, SAC, Pointer Networks, routing, or CH
retraining. Those changes add scope and do not solve the identified feature-use
failure.

## 6. Replacement Phase 2B gate

Freeze this gate before fine-tuned results are inspected.

### 6.1 Probe construction

Use at least 1,000 matched development-only counterfactuals stratified by:

- budgets 8, 12, 16, 20, 24;
- queue caps 1, 2, 3;
- low/middle/high residual energy;
- scarce clusters where alive demand exceeds budget and non-scarce clusters;
- all solar states and all thermal states;
- early, middle, and late episode observations.

### 6.2 Gate conditions

All conditions must pass:

1. **Marginal ordering:** paired median
   (d_{i,k}(s_i^+)-d_{i,k}(s_i^-)>0), with a one-sided 95% bootstrap
   confidence interval excluding zero.
2. **Material action response:** the probed node's projected allocation changes
   in at least 10% of scarcity counterfactuals. This threshold is a declared
   engineering materiality criterion, not a universal theorem.
3. **Directionality:** among non-tied changed allocations, at least 90% give
   weakly more service to the higher-(\rho) state. Report an exact binomial
   confidence interval.
4. **Feature ablation:** permuting the hybrid trajectory block causes a
   statistically detectable change in marginal values and projected actions;
   energy/queue controls are reported alongside it.
5. **No collapse:** zero-action and delivery metrics remain within declared
   non-collapse ranges.
6. **Concavity:** hard projection yields zero inference-time violations; report
   the pre-projection violation rate rather than hiding it.

The action-response threshold must be evaluated in scarcity states. Reset
states with queue cap one are insufficient for a multi-slot policy.

## 7. New evaluation protocol

The algorithm change requires a dated preregistration amendment. The earlier
partially executed Phase 4 seeds have already been exposed, so use a fresh,
hashed paired seed list for the revised algorithm. Do not silently reuse or
discard unfavorable old rows.

### 7.1 Main policies

Minimum defensible comparison set:

1. static equal TDMA;
2. energy-proportional;
3. harvest-proportional;
4. S2A2MAC-style adaptation;
5. FFSS-style adaptation;
6. original unregularized HTA-MAC checkpoints;
7. structure-constrained HTA-MAC;
8. independent-DQN structure ablation at budget 12.

The original checkpoint is now a scientifically valuable ablation: it isolates
whether the structural repair changes trajectory use and network behavior.

SHR-TDMA and Ge et al.'s cooperative RL should appear in a mechanism-level
comparison table. Implement them only if their full mechanisms can be mapped
faithfully to the round simulator; otherwise do not present a misleading
numeric “reimplementation.”

### 7.2 Main statistics

- 30 paired seeds per declared arm;
- median and IQR for descriptive reporting;
- paired Wilcoxon signed-rank tests;
- Holm correction over the frozen primary family;
- matched rank-biserial effect size and bootstrap confidence interval;
- right-censor lifetime events at common schedule coverage;
- Kaplan-Meier or restricted event-free-time summaries when FND/HND are
  censored;
- archive every raw trial, failure, and seed.

### 7.3 Essential ablations

If time is severe, keep these four and drop cosmetic sweeps:

1. unregularized versus structure-constrained loss;
2. hybrid trajectory block versus trajectory block permuted/removed;
3. solar-only versus solar+thermal;
4. full-data-slot idle energy versus 100-bit header sensitivity.

The five-budget frontier remains important because the completed development
results show lifetime and delivery move in opposite directions. Never report
one post-hoc “best” budget.

## 8. Revised contribution claims

Do not use the old C1-C4 wording. Conditional on the new experiments, use:

> **C1:** An idle-aware intra-cluster MAC formulation under a frozen exogenous
> CH schedule that assigns discrete per-node slot counts from separate solar
> and thermal HMM state-transition features.

> **C2:** A structure-constrained Branching Dueling distributional Q method
> combining physically normalized trajectory inputs, counterfactual marginal
> ordering, and diminishing-return branch regularization.

> **C3:** An exact greedy budget-projection result for the learned
> separable-concave slot-utility surrogate, with linear branch output growth and
> hard feasibility at inference.

> **C4:** A paired empirical characterization of the lifetime-delivery-fairness
> frontier, including trajectory, source, architecture, and idle-accounting
> ablations.

Do not claim strict lifetime dominance unless Phase 5 actually proves and
validates the assumptions. Do not claim a true Bayesian “posterior”; the
current features are state-conditioned transition probability vectors.

## 9. Paper narrative

Recommended title:

> **Structure-Constrained Trajectory-Aware Branching Q-Learning for Idle-Aware
> Intra-Cluster Scheduling in Hybrid Energy-Harvesting Sensor Networks**

Recommended central argument:

1. HEART-CH chooses CHs but leaves intra-cluster slots static.
2. Explicit idle listening exposes the cost of keeping non-transmitting members
   awake.
3. Generic BDQ trains stably but can ignore small, semantically critical
   trajectory inputs—demonstrated by the original ablation.
4. HTA-MAC inserts physically justified structure into representation, marginal
   values, and constrained projection.
5. The resulting method is evaluated as a Pareto scheduler, not claimed to win
   every metric simultaneously.

This story is stronger than hiding the failed probe. The failure becomes the
motivation for a measurable algorithmic advance.

## 10. Venue decision

### IEEE TGCN

Attempt TGCN only if the revised algorithm passes the trajectory gate, beats or
meaningfully shifts the Pareto frontier against strong baselines, and the
mathematical projection result is presented rigorously. TGCN remains a good
scope match, but the synthetic thermal model and absence of a real thermal
trace are significant review risks.

### IEEE Sensors Journal

This is the recommended primary venue when time is the dominant constraint.
Its official topic list includes sensor networks, sensor power systems,
energy harvesting, and machine learning for sensor data/systems
([scope](https://ieee-sensors.org/ieee-sensors-journal/)). The journal reports a
median submission-to-ePublication time of 8.8 weeks, but this is not an
acceptance promise.

### IEEE Access

Use as fallback if rapid handling is more important than selectivity. IEEE
Access states a typical 4-6 week submission-to-publication target and about a
20% acceptance rate, while still requiring a clear state-of-the-art advance
([review process](https://ieeeaccess.ieee.org/authors/stages-of-peer-review/)).

The present unmodified model should not be submitted to any of these venues.

## 11. Time-critical execution schedule

### First 6 hours

- freeze and hash this rescue specification;
- implement deterministic feature normalization;
- implement marginal/concavity diagnostics on saved checkpoints;
- generate mid-episode stratified counterfactual probe bank;
- verify old checkpoint reproduction before fine-tuning.

### Next 6-12 hours

- implement `L_traj`, `L_conc`, and next-harvest auxiliary head;
- unit-test counterfactual isolation and gradients;
- fine-tune budget 12 for the three optimizer seeds for 50 episodes;
- run the new gate without looking at Phase 4 metrics.

### Go/no-go checkpoint

- If trajectory and directionality improve materially, extend to 100-150
  episodes and all five budgets.
- If they remain weak, add the local trajectory residual head and repeat only
  the budget-12 development screen.
- If the residual-head version also fails, stop claiming trajectory-aware RL
  and pivot the paper to an analytical harvest-proportional scheduler. Do not
  spend more compute disguising a negative result.

### Following 24 hours

- fine-tune all 15 shared checkpoints and the three independent checkpoints;
- archive a new versioned registry;
- freeze the new evaluation seed list and statistical family.

### Following 2-4 days

- run paired policy evaluation;
- run the four essential ablations;
- generate tables and plots directly from archived CSVs;
- write Methods and Results from actual outputs.

Compute time is uncertain until a 50-episode fine-tuning benchmark is run. A
100-episode warm-start campaign should be much shorter than the original
500-episode 18.68 GPU-hour sweep, but no runtime should be promised before the
benchmark.

## 12. Reviewer-risk checklist

The manuscript is not ready until every answer is “yes”:

- Does a trajectory intervention materially change marginal slot values?
- Does it change projected actions in scarce states in the expected direction?
- Does hybrid outperform or at least measurably differ from solar-only?
- Is the thermal source explicitly labeled synthetic and not dataset-trained?
- Is greedy optimality stated only for the concave learned surrogate?
- Are all policies evaluated in the identical idle-aware environment?
- Are censoring and incomplete deaths handled statistically?
- Are the original 1191.3 and reproduced 1100.6 baselines distinguished?
- Are newer competitors discussed without cross-environment numeric claims?
- Does every table cell trace to an archived raw file?

## 13. Final research decision

The fastest credible publication route is not a wholesale new model. It is a
warm-started, structure-aware correction that turns the discovered weakness
into the paper's algorithmic motivation:

\[
\text{stable BDQ but ignored trajectory}
\;\longrightarrow\;
\text{normalized + counterfactual + concave BDQ}
\;\longrightarrow\;
\text{exact feasible projection and measurable trajectory use}.
\]

If the corrected model passes the prespecified mechanism gate and produces a
useful paired Pareto frontier, the paper has a defensible contribution. If it
does not, the honest publishable pivot is an analytical/heuristic idle-aware
MAC paper—not a claim that the current neural policy is trajectory-aware.
