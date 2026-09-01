# HTA-MAC delivery and packets/J improvement plan

**Prepared:** 1 September 2026  
**Status:** Plan only; no new training is authorized by this document.  
**Objective:** Produce a new HTA-MAC candidate that beats strong reference
policies on delivery and packets/J under one common simulator while retaining
hard feasibility and acceptable fairness, staleness, and lifetime.

## 1. Scientific target

Published percentages from different media, simulators, horizons, and energy
models cannot form a valid leaderboard. The optimization target is therefore a
same-simulator result against:

1. author-constructed queue-cap-feasible residual-energy greedy;
2. author-constructed online dual-ascent QoS;
3. a recognized constrained-RL baseline added before final evaluation,
   preferably PPO-Lagrangian or guard-enhanced PPO.

The corrected 100-node reference values to exceed are:

| Metric | Current HTA-MAC | Strongest current reference |
|---|---:|---:|
| Delivery | 0.42770 | 0.44591 |
| Packets/J | 225.77 | 242.36 |
| Stale ratio | 0.02476 | 0.01700 |
| Jain fairness | 0.94511 | 0.98081 |
| Restricted survival | 128.28 | 149.32 |

The primary development target is deliberately harder than a numerical tie:

- delivery >= 0.450;
- packets/J >= 244.8;
- stale ratio <= 0.035;
- Jain fairness >= 0.92;
- restricted survival >= 128 rounds;
- zero action-budget, queue-cap, or dead-node feasibility violations.

The delivery and packets/J targets are approximately 1% above the strongest
current reference. Fairness and staleness may move from their current values,
but only inside explicitly frozen tolerances.

## 2. Why previous improvement attempts did not solve it

The current residual controller removes service to stay near a fixed delivery
band. It does not sufficiently adapt when demand, node count, or contention
changes. Earlier workload-conditioned adapters learned nonzero workload weights,
yet reduced delivery on fresh seeds and usually reduced packets/J. This rules
out the simple explanation that the model merely lacked two global workload
features.

The corrected residual-energy heuristic simultaneously obtains more delivery
and more packets/J. Therefore, the gap is not only excessive transmission. The
current policy is selecting an inferior mix of service opportunities: it either
under-serves useful queued packets, spends slots on high incremental-energy
transmissions, or fails to account for the member-TX plus CH-forwarding cost of
each successful packet.

Another ungated C51 fine-tune or reward-weight sweep is not the next experiment.

## 3. Proposed method: constrained marginal-utility scheduling

### 3.1 Per-slot utility and energy cost

For every eligible member `i` and its next possible slot `k`, estimate:

- probability of successful packet service;
- packets expected to be delivered;
- stale packets expected to be avoided;
- fairness deficit reduction;
- member-to-CH transmit energy;
- additional CH receive/aggregate/forward energy;
- depleted-CH risk;
- diminishing return from assigning another slot to the same member.

Define a marginal score of the form

`benefit(i,k) - lambda_E * joules(i,k) - lambda_CH * depleted_CH_risk(i,k)`,

where benefit combines expected delivery, deadline urgency, and a fairness dual.
Allocate slots greedily by marginal score under the exact common budget and
queue caps. This is a MaxWeight/drift-plus-penalty style controller, not a new
metric invented after seeing results.

### 3.2 Learn the residual, not the energy law

Use the analytical marginal-utility allocator as a teacher and learn only a
permutation-equivariant residual correction. The analytical core provides a
strong energy-aware inductive bias; the learned residual captures nonlinear
queue, harvesting, role-risk, and future-value effects.

The model remains MAC-only. It must not change CH selection, routing, harvesting,
traffic generation, radio equations, or the evaluation horizon.

### 3.3 Required equivariant features

Add only branch-local or permutation-safe set-context features:

- normalized queue and queue-cap headroom;
- deadline slack and expiring-packet fraction;
- member-to-CH distance and estimated member-TX joules;
- scheduled CH residual energy percentile;
- estimated CH receive/aggregate/forward joules per served packet;
- predicted success probability;
- residual-energy and service-deficit percentiles within the active cluster;
- normalized offered load and current budget pressure.

Every feature must pass permutation-equivariance, padding, scale, and finite-value
tests before any training.

### 3.4 Constrained learning objective

Use a two-stage procedure:

1. **Teacher imitation:** train the shared scorer to reproduce marginal teacher
   ordering and feasible allocations over a multi-scenario state corpus.
2. **Constrained fine-tuning:** use PPO-Lagrangian or an equivalent dual-update
   objective to maximize delivered packets per joule while enforcing staleness,
   fairness, and survival constraints.

Do not combine delivery and packets/J into an undocumented weighted score.
Report the primal objective and every constraint separately. The exact-budget
projection remains outside the policy network and is tested exhaustively.

## 4. Experimental sequence and stop gates

### Phase A — no-training mechanism audit

Instrument the current policy, residual-energy greedy, and dual-ascent on a new
development cohort. For every allocated slot, record:

- whether it served a real queued packet;
- whether delivery succeeded;
- member-TX energy;
- CH receive/aggregate/forward energy;
- stale avoidance;
- marginal packets/J;
- node and CH residual energy;
- unused feasible demand and unused budget.

**Gate A:** continue only if at least 80% of the delivery/efficiency gap can be
assigned to observable selection, under-service, or marginal-energy mechanisms.
If it cannot, inspect simulator accounting before changing the model.

### Phase B — deterministic teacher

Implement the constrained marginal-utility allocator and evaluate it without
learning on new development seeds.

**Gate B:** the teacher must reach delivery >= 0.450 and packets/J >= 244.8 while
meeting the frozen secondary constraints. If the teacher fails, a neural student
cannot be expected to solve the same target reliably; stop and report that the
frontier is structural under the current MAC scope.

### Phase C — architecture and imitation smoke test

Add the new features and shared marginal-residual head. Run:

- permutation tests over random active-node permutations;
- padding and variable-node-count tests from 20 through 300 nodes;
- exact-budget and queue-cap property tests;
- teacher-order agreement on unseen states;
- a short CPU smoke run that is not used for model selection.

**Gate C:** random-permutation inverse agreement >= 0.99, maximum marginal-Q
error <= a predeclared same-platform tolerance, zero feasibility failures, and
teacher top-choice agreement >= 0.95.

### Phase D — bounded candidate screen

Train three optimizer replicas over a multi-scenario curriculum containing node
counts 20, 50, 100, 150, 200, 250, and 300; low/reference/high traffic; harvest
scales; battery scales; field scales; and trace replay. Use development seeds
that have never been used for confirmation or manuscript reporting.

Evaluate checkpoints at fixed intervals and select using a lexicographic rule:

1. all feasibility and secondary constraints pass;
2. highest lower confidence bound for packets/J;
3. highest lower confidence bound for delivery;
4. no post-hoc preference for lifetime, fairness, or a favorable seed.

**Gate D:** at least two of three replicas must pass the primary targets on the
development cohort. Otherwise stop; do not train longer automatically.

### Phase E — recognized constrained-RL baseline

Train PPO-Lagrangian or guard-enhanced PPO with the same observations, action
caps, projection, scenarios, interactions, and optimizer-replica count. This
baseline is mandatory if the paper claims an advance in constrained RL.

**Gate E:** HTA-MAC must be non-inferior to the recognized baseline on delivery
and packets/J within margins frozen before running, and must retain at least one
significant advantage such as scale transfer, inference cost, staleness, or
fairness.

### Phase F — untouched confirmation

Freeze one candidate, all baseline implementations, endpoints, inference unit,
confidence intervals, effect sizes, multiple-testing family, and stop rules.
Use a fresh 20-seed cohort. Previously opened confirmation seeds must not be
reused for tuning or for a nominally new confirmation.

The publishable success condition is:

- both delivery and packets/J improvements over residual-energy greedy have
  paired 95% confidence intervals entirely above zero;
- the same metrics are non-inferior to the recognized constrained-RL baseline;
- fairness, staleness, and survival constraints pass;
- conclusions remain valid under at least the reference, high-traffic, 300-node,
  and external-trace scenarios.

If any primary gate fails, publish the measured Pareto frontier or retain the
current model. Never redefine the metric, change the comparison cohort, or
select the least-bad candidate after opening confirmation data.

## 5. Compute discipline

The sequence deliberately spends little compute until the deterministic teacher
proves that the target is reachable:

1. mechanism audit: CPU only;
2. deterministic teacher: CPU only and parallel over seeds;
3. architecture/property tests: CPU only;
4. bounded three-replica training: GPU only after Gates A--C pass;
5. recognized baseline: GPU only after a candidate passes Gate D;
6. final 20-seed evaluation: CPU parallelism, with no training.

High CPU utilization is useful for independent simulator rollouts, but not for
neural minibatch training when a GPU is available. Parallelism must be across
independent environments or seeds, not by creating correlated pseudo-replicates.

## 6. Expected outcome

This plan has a credible route to closing the current 4.1% delivery and 6.8%
packets/J gaps because it targets the marginal service-energy mechanism that the
current fixed-band residual does not model. It does not guarantee success. Gate
B is the decisive low-cost test: if a full-information constrained teacher
cannot beat both thresholds, no amount of confidence or additional C51 epochs
should be spent claiming that the neural policy will do so.
