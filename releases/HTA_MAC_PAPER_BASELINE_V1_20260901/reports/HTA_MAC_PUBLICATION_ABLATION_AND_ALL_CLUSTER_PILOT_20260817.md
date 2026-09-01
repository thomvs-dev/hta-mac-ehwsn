# HTA-MAC Publication Ablation and All-Cluster Pilot

**Date:** 17 August 2026  
**Status:** implementation complete; smoke and five-run development pilot complete; full retraining and independent evaluation pending

## Why this extension was required

The manuscript previously evaluated one scheduled cluster per rollout while
non-target clusters used equal allocation. That design isolated the MAC effect,
but it could not establish the network-wide effect of deploying the policy in
every cluster. It also evaluated one retained training lineage and lacked
component ablations for CH context and auxiliary losses.

This extension adds:

1. simultaneous control of every active scheduled cluster in one shared
   physical network trajectory;
2. FND, HND, and LND event tracking with explicit censoring;
3. matched retraining ablations over three optimizer initializations;
4. same-checkpoint mechanism interventions for CH context and invariant set
   context;
5. an exact dynamic-programming allocation solver to measure greedy projection
   regret under non-concave learned values; and
6. per-cluster and per-network-round decision latency.

CH selection, association, and routing remain exogenous. Each cluster retains
its own 24-slot budget.

## Implemented ablation matrix

The matched retraining protocol contains three optimizer initializations for
each arm:

| Arm | CH context | Trajectory-order loss | Concavity loss |
|---|---:|---:|---:|
| Full | on | 1.0 | 0.1 |
| No CH context | zeroed during training and inference | 1.0 | 0.1 |
| No trajectory loss | on | 0.0 | 0.1 |
| No concavity loss | on | 1.0 | 0.0 |
| No auxiliary losses | on | 0.0 | 0.0 |

The training smoke executed all 15 lineages. Every process persisted a
checkpoint and returned the expected smoke-only status; no traceback or CLI
argument failure occurred. Full 100-episode retraining has not yet been run.

## All-cluster smoke

The two-scenario, 20-round smoke completed all ten policy/scenario tasks with:

- zero per-node cap violations;
- zero per-cluster budget violations; and
- one shared network-wide action per round.

This short smoke was used only for wiring validation and is not a performance
result.

## Five-run full-horizon development pilot

The pilot used five new development realizations, a 3000-round cap, and three
conditions: reference 100 nodes, 20 nodes, and high traffic. Twenty independent
evaluation realizations remain unused.

### Reference, 100 nodes

| Policy | Delivery | FND | HND | LND/RMST cap | Packets/J |
|---|---:|---:|---:|---:|---:|
| HTA-MAC | 0.6416 | 120.6 | 166.4 | 3000 censored | 345.96 |
| No CH context intervention | 0.6504 | 122.0 | 168.6 | 3000 censored | 349.15 |
| No set context intervention | 0.6680 | 122.2 | 168.4 | 3000 censored | 356.95 |
| Energy-proportional | 0.6484 | 156.4 | 167.4 | 3000 censored | 335.71 |
| Online primal-dual | 0.6557 | 133.2 | 165.6 | 3000 censored | 336.76 |

The full policy is more energy-efficient than both baselines, but its delivery
is lower and FND occurs earlier. HND is within about one round of
energy-proportional and is slightly later than primal-dual. The difference
between FND and HND shows that the earliest vulnerable node is not representative
of whole-network depletion.

### Twenty nodes

| Policy | Delivery | FND | HND | LND/RMST cap | Packets/J |
|---|---:|---:|---:|---:|---:|
| HTA-MAC | 0.8524 | 152.4 | 178.0 | 3000 censored | 389.26 |
| Energy-proportional | 0.8791 | 156.2 | 184.6 | 3000 censored | 394.93 |
| Online primal-dual | 0.8817 | 155.0 | 181.0 | 3000 censored | 393.26 |

The earlier one-cluster 20-node Pareto advantage does not reproduce in this
network-wide development pilot. This result must be reported if it remains in
the independent evaluation.

### High traffic

| Policy | Delivery | FND | HND | LND/RMST cap | Packets/J |
|---|---:|---:|---:|---:|---:|
| HTA-MAC | **0.4013** | 105.2 | 127.0 | 3000 censored | **340.31** |
| Energy-proportional | 0.3642 | **124.8** | 128.8 | 3000 censored | 293.03 |
| Online primal-dual | 0.3574 | 113.2 | **129.8** | 3000 censored | 316.38 |

Under high load, HTA-MAC improves delivery by 0.0370 over
energy-proportional and by 0.0439 over primal-dual. It also improves packets/J
by 47.28 and 23.93 respectively, but advances FND by 19.6 and 8.0 rounds. This
is the clearest candidate contribution: service and energy-efficiency gains in
a congested regime, with an explicit weakest-node lifetime cost.

## Projection optimality

For the full policy, the greedy marginal projector matched the exact separable-Q
optimum in approximately 99.8--99.9% of decisions across the pilot scenarios.
Mean objective regret was below `9e-5`; maximum observed regret was below 0.15.
The hard feasibility claim is exact, while global Q-optimality is now measured
rather than assumed.

Removing set context increased projection mismatch and regret, reaching a
maximum regret near 0.96 under high traffic. This is a mechanism diagnostic,
not a trained architecture ablation.

## Interpretation and next decision

The all-cluster result changes the paper's strongest narrative:

- The former 20-node Pareto-superiority statement is not yet network-wide.
- HTA-MAC's strongest network-wide operating region appears to be high traffic,
  not the reference or sparse regime.
- FND and HND tell different stories; both must be reported.
- The set and CH contexts are not automatically beneficial in inference-only
  interventions, so matched retraining is necessary before making component
  claims.
- LND is right-censored and must not be reported as 3000-round observed
  lifetime.

The correct next sequence is:

1. run all 15 matched 100-episode training lineages;
2. evaluate every lineage without best-lineage selection;
3. use training-lineage means as the ablation inference unit;
4. only then open the 20-run all-cluster evaluation cohort; and
5. revise the paper around the supported operating regime rather than forcing
   the previous one-cluster conclusion.

Local profiling showed that environment stepping is effectively single-core per
lineage. Full ablation training therefore uses 15 concurrent single-threaded
lineages, occupying up to 15 logical CPUs without changing any learning
parameter, seed, or statistical evidence unit. Expected wall time is close to
one 100-episode training batch rather than five sequential batches.

The same profiling result applies to all-cluster rollouts. Evaluation therefore
uses 16 single-threaded workers rather than 8 workers with 2 ineffective math
threads; scenario definitions, seeds, checkpoints, and estimands are unchanged.

## Full matched-training outcome (17 August 2026)

The 15-lineage matrix completed in 72.04 minutes using 15 concurrent
single-threaded workers. Initial measured host utilization was 95--97%. All
checkpoints and logs were persisted, but every process returned the declared
gate-failure code: `all_processes_completed=true` and
`all_training_gates_passed=false`. The training manifest SHA-256 is
`7127936E438965C84A2282680408957B5D7B2C5524783AF54CABB2A78ED75EE7`.

This is not a successful publication ablation. No lineage passed the reward
convergence test. Four of fifteen lineages collapsed to all-sleep: one full,
one no-CH-context, and two no-concavity-loss runs. Only the full,
no-CH-context, and no-concavity variants produced one QoS-passing seed each;
the other twelve lineage results failed QoS. Mean greedy packets and zero-action
fractions across the three seeds were: full 1444.7/0.36, no CH context
1448.0/0.37, no trajectory loss 1801.7/0.14, no concavity loss 775.8/0.68,
and no auxiliary losses 1747.7/0.23.

The principal diagnosis is seed-sensitive instability after categorical-output
reinitialization, compounded by an overactive QoS delivery penalty in failing
lineages. The mean active delivery-penalty fraction reached 0.54--0.59 in the
collapsed examples despite a declared target band of 0.02--0.10. The 100-episode
budget also did not stabilize reward: all 15 last-half versus first-half
relative-change tests failed. Concavity appears protective but insufficient;
removing it increased collapse from one of three full lineages to two of three.
Trajectory loss is not the sole cause of failure, and removing both auxiliary
losses prevented literal collapse but did not yield stability or QoS passage.

The sealed final evaluation must therefore remain unopened. The next defensible
experiment is a development-only repair of initialization and QoS-controller
geometry, followed by a small multi-seed convergence probe. Longer training is
justified only after that probe eliminates all-sleep collapse and brings the
active penalty fraction inside its preregistered band; simply selecting the
best current seed is prohibited.
The independent all-cluster evaluation is expected to require roughly one hour.

## Claim boundary

These are five-run development-pilot results. They are not final paper results,
do not authorize model selection on the independent cohort, and do not support
universal lifetime or network-wide superiority claims.
