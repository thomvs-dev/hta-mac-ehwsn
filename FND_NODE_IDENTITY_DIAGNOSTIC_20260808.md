# B16 FND Node-Identity Diagnostic

**Date:** 8 August 2026  
**Scope:** selected HTA-MAC seed 7399 versus energy-proportional, development seeds 2400–2404, 3,000-round B16 profile  
**Claim boundary:** mechanism diagnostic only; not a superiority test and not confirmation

## Result

The FND clustering is mostly caused by a shared, seed-specific vulnerable physical node, but HTA-MAC consistently reaches that failure earlier.

- All 10 replayed FND rounds exactly match the archived long-horizon CSV.
- HTA-MAC and energy-proportional lose the identical physical node in 4/5 paired seeds.
- The shared node is different in each environment seed; there is no single globally defective node ID.
- HTA-MAC reaches FND 10–48 rounds earlier in every paired seed; the median paired difference is -39 rounds.
- The dying node has been scheduled as CH for approximately 20% of rounds under both policies. The finding is therefore not explained by an anomalously high CH-selection frequency.
- HTA-MAC's dying node is serving as CH on the exact FND round in 5/5 trials. The energy-proportional dying node is CH on its FND round in 3/5 trials.

| Seed | HTA FND | HTA node | Energy-proportional FND | EP node | Same node? | HTA node CH at FND? | EP node CH at FND? |
|---:|---:|---:|---:|---:|---|---|---|
| 2400 | 1,039 | 48 | 1,078 | 48 | yes | yes | yes |
| 2401 | 1,027 | 60 | 1,075 | 60 | yes | yes | yes |
| 2402 | 1,080 | 49 | 1,090 | 49 | yes | yes | no |
| 2403 | 1,108 | 68 | 1,132 | 68 | yes | yes | no |
| 2404 | 1,081 | 42 | 1,127 | 46 | no | yes | yes |

## Mechanistic interpretation

The earlier report's objective/horizon diagnosis remains valid: the training horizon is 300 rounds, the death term fires zero times in 450,000 training steps, and the first evaluation death occurs around round 1,080. The policy is never trained in the state regime underlying the lifetime endpoint.

The new evidence narrows that diagnosis. FND is not freely movable by MAC allocation alone. In 80% of paired trials, both aggressive-service policies ultimately exhaust the same seed-specific node under the same exogenous schedule and harvesting realization. The frozen schedule/topology/harvest process therefore defines a shared vulnerability ceiling. HTA-MAC's allocation choices affect **when** that node crosses zero energy, but usually not **which** node is structurally most exposed.

This is a mixed causal result:

1. **Structural component:** the exogenous CH schedule, node geometry, and harvesting realization identify a weak node shared across aggressive policies.
2. **MAC-controlled component:** HTA-MAC reaches that shared failure 10–48 rounds earlier and always crosses the boundary while the dying node is acting as CH. Higher successful service into an energy-poor scheduled CH can accelerate its receive/aggregate/base-station transmission cost.
3. **Scope limit:** HTA-MAC cannot change the frozen CH assignment. A MAC-only lifetime fix must throttle or redistribute member service when a scheduled CH has high depletion risk, accepting an explicit delivery/lifetime trade-off. Changing the CH schedule would be a different intervention outside the frozen HEART-CH MAC-only scope.

The data do not support the stronger statement that HTA-MAC is simply “the most aggressive spender.” At 300 rounds it allocates fewer nominal slots than energy-proportional, and empty allocations cost no idle energy in B16. The relevant mechanism is successful packet service and CH forwarding cost, not nominal slot count alone.

## Consequence for Step 3

Step 3 should proceed, but its success criterion must be realistic and role-conditioned:

- preserve a frozen minimum delivery/stale-service floor;
- expose training to FND or a validated pre-death risk regime;
- condition the risk surrogate on scheduled-CH role, lower-tail residual energy, predicted incoming service, BS distance, and harvest forecast;
- log cumulative energy separately for member transmission and CH receive/aggregate/BS-forwarding roles;
- compare the FND node identity and failure-role transition after retraining;
- do not expect a large universal FND gain without changing the upstream CH schedule;
- treat any CH-policy modification as a separate experiment rather than expanding HTA-MAC scope silently.

The strongest defensible target is not “beat every conservative policy on FND.” It is: **delay the shared aggressive-policy weak-node failure while maintaining the preregistered global service floor.**

## Evidence artifacts

- Aggregate JSON: `outputs/diagnostics/paper_aligned_b16_qos_repaired_fnd_identity_20260808.json`
- Reusable replay audit: `experiments/audit_phase3_fnd_node_identity.py`
- Per-seed artifacts: `outputs/diagnostics/fnd_identity_seed2400.json` through `fnd_identity_seed2404.json`
- Tests: `validation/test_fnd_node_identity_audit.py`
