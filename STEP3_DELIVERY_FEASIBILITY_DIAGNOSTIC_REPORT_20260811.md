# Step 3 Delivery-Feasibility Diagnostic Report

**Date:** 11 August 2026  
**Scope:** development-only, no-learning diagnostic  
**Frozen setting:** seed 2400, 20 target ranks, 1,200 rounds, budget 16,
unchanged exogenous CH schedule

## Decision

The delivery floor is **structurally reachable** under the frozen environment.
The fairness-aware budget-filling oracle passed delivery on 20/20 pairs and
joint QoS on 18/20 pairs, exactly meeting the registered 90% requirement.

This authorizes design of another bounded development candidate. It does not
authorize 500-episode training, held-out evaluation, model selection, or a
publication claim.

## Comparison

| Metric | Trained greedy | Budget-fill oracle |
|---|---:|---:|
| Joint QoS passes | 0/20 | 18/20 |
| Delivery passes | 0/20 | 20/20 |
| Stale passes | 20/20 | 20/20 |
| Fairness passes | 18/20 | 18/20 |
| Macro delivery ratio | 0.486470 | 0.661942 |
| Micro delivery ratio | 0.486233 | 0.660705 |
| Mean fairness | 0.827900 | 0.854412 |
| Mean FND-free rounds | 1170.75 | 1120.45 |

The oracle proves feasibility but costs 50.30 mean FND-free rounds. It is an
upper-service diagnostic, not a suitable final MAC policy.

## Allocation-loss decomposition

Across the trained-policy rollouts:

- budget-feasible delivery opportunities: 228,711 packets;
- executed delivery: 175,571 packets;
- opportunity utilization: 76.765%;
- total service gap: 53,140 packets;
- pre-projection/Q-choice gap: 41,159 packets (77.454% of the gap);
- projection reduction: 11,981 packets (22.546% of the gap).

The main defect is therefore conservative learned Q preferences before budget
projection, with a smaller but material projection contribution. Global unused
slot totals must not be read alone because they also include empty-target and
backlog-exhausted rounds; `budget_feasible_delivery - executed_delivery` is the
valid opportunity-conditioned gap.

Nearly every evaluated round changes scheduled membership or CH identity.
Consequently, almost all service gaps occur after a schedule transition, but
this correlation is not evidence that transitions cause the gap. It confirms
that a schedule-matched, identity-preserving policy is required.

## Energy and lifetime mechanism

Relative to the trained policy, the oracle used:

- 21.64% more member-TX energy;
- 18.34% more CH-forwarding energy;
- 50.30 fewer mean FND-free rounds.

The environment has no separate CH packet-capacity limit: a CH can forward all
received target packets in a round, and forwarding is limited indirectly by
energy/death. The next policy must therefore recover selected unused service
opportunities while conditioning on scheduled-CH reserve and forwarding cost.

## Fairness edge case

The oracle's two joint failures were target ranks 5 and 11. Both passed
delivery and stale thresholds but ended with fairness reported as zero. This
is associated with schedule states having no active target members and should
be audited as a terminal fairness-definition edge case before treating those
two failures as allocation failures.

## Next bounded candidate

Implement a **risk-gated budget-completion layer**, without changing the CH
schedule or network environment:

1. retain the learned projected action as the primary decision;
2. compute unused opportunity only over alive target members with feasible
   backlog;
3. add slots greedily only when the scheduled CH's predicted post-forwarding
   reserve remains above a frozen safety floor;
4. use marginal Q for ranking, but allow a small frozen negative-gain tolerance
   so near-zero Q errors do not suppress feasible service;
5. cap additional service by an explicit per-round completion fraction;
6. retain trajectory-order loss, concavity loss, QoS constraint, and CH-risk
   term;
7. run a no-learning parameter sweep first and authorize at most one new
   100-episode bounded probe only if it reaches at least 18/20 QoS pairs while
   keeping mean FND within a predeclared non-inferiority margin.

Recommended development grid:

- CH post-forwarding reserve floor: 0.10, 0.15, 0.20;
- budget-completion fraction: 0.25, 0.50, 0.75;
- marginal-Q tolerance: 0.00 and a scale-normalized small negative tolerance;
- FND non-inferiority margin: no worse than 1% of the 1,200-round horizon
  relative to the current trained checkpoint.

Do not train any grid point. Evaluate these settings without learning, freeze
the single candidate only if all predeclared gates pass, then run one bounded
100-episode probe.

## Artifacts

- `experiments/diagnose_step3_delivery_feasibility.py`
- `validation/test_step3_delivery_feasibility.py`
- `outputs/audits/STEP3_DELIVERY_FEASIBILITY_DIAGNOSTIC_20260811.json`
- `outputs/local_cpu_export/step3_v3_bounded_100ep_seed5599_cpu18/stability_episode_100.pt`

