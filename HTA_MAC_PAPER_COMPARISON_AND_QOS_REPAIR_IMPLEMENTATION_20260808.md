# HTA-MAC paper comparison and post-repair implementation report

**Prepared:** 8 August 2026  
**Scope:** exploratory paper-aligned B16 development branch only  
**Evidence rule:** measured HTA-MAC results are separated from literature-reported results; cross-study percentages are contextual and are not head-to-head rankings.

## Executive conclusion

HTA-MAC's defensible advantage is not yet lifetime superiority. In the completed 300-round development experiment, it delivered 29,980 of 30,100 generated packets (median delivery ratio 0.996013), had zero median stale drops, global service fairness 0.999998, and kept all 100 nodes alive. It improved median throughput over static equal TDMA by 25.09%, but used 22.04% more energy, yielding only a 2.60% packets/J improvement. It was effectively tied with the energy-proportional diagnostic and slightly less energy-efficient. All FND/HND observations were right-censored at 300 rounds.

The strongest current differentiators are therefore:

1. hard per-cluster B16 and per-node slot-cap feasibility;
2. an explicitly exogenous CH schedule, so the learned contribution remains MAC-only;
3. a permutation-equivariant shared policy with architecture, projection, stability, and C51-support audits;
4. near-lossless short-horizon global service under a 100-node terrestrial solar profile; and
5. broader reporting of delivery, stale drops, fairness, energy, efficiency, and censor-aware lifetime rather than a throughput-only claim.

The prior QoS training ratio was invalid because rotating target clusters could service queued packets generated outside the current denominator. That branch is retained as pre-repair evidence, but it must not support constrained-RL claims. The repaired implementation trains fresh lineages using a consistent target-backlog service cohort and then evaluates global end-to-end metrics separately.

## What comparable papers report

| Work | Environment and intervention | Reported result | What can and cannot be compared |
|---|---|---|---|
| Hasani et al., *Scientific Reports* (2025) | DQN-based energy-aware WSN control; paper-aligned HTA profile uses its 100-node, 0.5 J terrestrial parameters | 11.79% throughput improvement; the paper also reports that packet loss rises as received packets/throughput rises | HTA's +25.09% vs static is numerically larger, but not a reproduction or common-code comparison. HTA's 0.996 global delivery and zero median stale drops are a useful additional quality check. [Primary source](https://www.nature.com/articles/s41598-025-14111-y) |
| Ge, Nan, and Guo, *International Journal of Distributed Sensor Networks* (2021) | Cooperative Q-learning/SARSA for duty cycle and transmission rate in 20/40/60-node clustered solar WSNs | Under battery degradation, throughput improves 16.6%-30.1% and dead-node count falls 21.0%-45.9% versus static; the paper also includes a random-search comparison | HTA's +25.09% lies within that paper's reported throughput-improvement band, but node count, simulator, horizon, traffic, energy, and actions differ. HTA has not yet shown dead-node improvement because no death occurred by round 300. [Primary source](https://journals.sagepub.com/doi/10.1177/15501477211007411) |
| Eris, Gul, and Boluk, *Sensors* (2024) | Cooperative RL slot-position selection in a 100-node clustered underwater EH network | Median FND is about 25 for all methods; HND 113/128/128 for no-EH/EH/RL TDMA; LND 355/442/498. RL receives about 11% more packets than TDMA-EH and captures 96% of available energy versus 56% | This paper demonstrates the lifetime endpoint discipline HTA still needs. Its underwater acoustic model is incompatible with the terrestrial radio profile, so absolute lifetimes must not be compared. [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC11487392/) |
| HENO-MAC (2024) | Hybrid solar-wind receiver duty-cycle adaptation with a realistic two-day GreenCastalia trace | Delay reductions up to 28.5% for all packets and 27.3% for highest-priority traffic | HTA currently lacks comparable priority-delay and real-trace evidence. This is a gap, not an HTA advantage. [Author manuscript](https://arxiv.org/abs/2401.00717) |
| FFSS/AFSS (2021) | Future-energy/future-data slot ordering with Hungarian assignment | Optimizes channel utilization by assigning qualified nodes to earlier slots | HTA chooses integer per-node slot counts under a hard frame budget; the current round-level simulator does not represent within-frame ordering, so the bundled FFSS policy is explicitly an adaptation. [Publisher abstract](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/cmu2.12243) |
| SHR-TDMA (2020) | Harvest-aware slot assignment under fixed/random interarrival energy | Minimizes waiting-for-slot delay using Hungarian assignment | HTA has richer learned state and hard budget projection, but no directly comparable within-frame delay endpoint. [Publisher abstract](https://ietresearch.onlinelibrary.wiley.com/doi/abs/10.1049/iet-com.2019.0977) |
| Structure-aware RL (2018) | Monotonic queue/battery policy for one delay-sensitive EH sensor | Exploits structural monotonicity to reduce RL search complexity | Supports a future monotonicity regularizer/audit, but it is a single-sensor delay problem and provides no network-lifetime comparator. [Author manuscript](https://arxiv.org/abs/1807.08315) |

## Where HTA-MAC is better now

### Better-supported claims

- **Throughput relative to its own static baseline:** +25.09%, larger than Hasani et al.'s reported 11.79% within their experiment and inside Ge et al.'s reported 16.6%-30.1% range. This is context, not proof of superiority across simulators.
- **Short-horizon traffic quality:** global delivery is 0.996013 with zero median stale drops, avoiding a throughput-only interpretation in which loss may also increase.
- **Feasibility assurance:** every learned action is projected into the B16 frame budget and per-node cap, with tests and audits rather than relying on reward penalties alone.
- **Identity handling:** the accepted architecture is permutation equivariant and uses a shared local head. This is stronger evidence than the rejected identity-specific Phase 2C branch.
- **Evaluation transparency:** energy cost, packets/J, fairness, censoring, and exact small-sample limitations are reported. With five paired seeds, the smallest possible nonzero two-sided exact Wilcoxon p-value is 0.0625.

### Not better or not yet known

- **Energy proportional:** HTA is tied in throughput and slightly worse in packets/J.
- **S2A2MAC-adapted:** HTA gains 29.62% throughput but consumes 34.64% more energy and is 3.67% worse in packets/J.
- **Lifetime:** unknown at 300 rounds because every FND/HND value is censored.
- **Delay, priority QoS, real traces, and hardware:** not established.
- **Third-party reproduction:** none of the paper-aligned comparisons is a reproduction of external source code or data.

## What was wrong in the previous training objective

The dynamic wrapper rotates the target cluster. The old controller accumulated:

- delivered packets from the target members' pre-existing queues; and
- newly generated packets only from the members alive in the current target round.

A queued packet could therefore be generated when its node was outside the target and delivered later when it entered the target. One recorded episode accumulated 2,806 delivered packets against 1,192 generated packets. The controller hid the mismatch by clipping the ratio to 1.0. Consequently, the old QoS multiplier behavior was not evidence of a valid end-to-end delivery constraint.

The queue-fairness name was also overloaded. Phase 2 target-cluster service fairness was approximately 0.703-0.708, while Phase 3 global network service fairness was approximately 0.999998. These are now separate metric scopes.

## Implemented repair

### Cohort-consistent training metric

At each target-cluster decision, the environment now records:

`target_packets_offered = sum(pre-service queue for alive current target members)`

Delivered packets are a subset of that exact same-step backlog. The new controller uses `episode_cumulative_target_backlog_service`, refuses `delivered > offered`, and no longer clips delivery or stale ratios. This metric is intentionally named a **service-opportunity ratio**; it is not presented as global end-to-end generated-packet delivery.

Legacy schema-2 configurations remain readable so old evidence is reproducible. The post-repair branch is schema 3 and stores its metric contract in controller evidence.

### Separated global evaluation gates

Training uses target-backlog service and target-cluster service fairness. Whole-network Phase 3 evaluation independently uses:

- global generated-packet delivery ratio;
- global stale-drop ratio;
- global network cumulative-service fairness; and
- censor-aware FND/HND summaries.

The 300-round development gate and the 3,000-round development evaluation are distinct. A 3,000-round censor time is never substituted for an unobserved FND/HND event.

### Fresh experiment and leakage controls

- Fresh optimizer seeds: **5399, 6399, 7399**.
- Development seeds: **2400-2404**.
- Reserved confirmation seeds: **3400-3404**, not run by the notebook.
- Registered held-out seeds **3100-3104** remain prohibited.
- Old 5299/6299/7299 checkpoints are not resumed or relabeled.

### Policy-distinctness audit

The new audit queries HTA-MAC and every comparator on identical states along an HTA-controlled trajectory. It reports exact action agreement, normalized L1 distance, active-set Jaccard similarity, mean slot use, and trajectory signatures. This distinguishes a genuinely different learned allocator from a policy that merely reproduces energy-proportional or another heuristic. It is a decision audit, not an outcome superiority test.

## Gated next sequence executed by the Colab notebook

1. Verify the bundle checksum and every manifest file.
2. Compile and run the complete validation suite, including the cohort regression tests.
3. Recalibrate C51 return scale using only development seeds and the repaired QoS controller.
4. Train three fresh 500-episode lineages.
5. Require curriculum, convergence, stability, permutation, projection, and C51-support gates.
6. Run 300-round whole-network paired development evaluation.
7. Select a candidate only if every development seed passes the separately named global gate.
8. Run the common-state action-distinctness audit.
9. Run a 3,000-round paired development evaluation and report censor-aware lifetime evidence.
10. Package all logs, checkpoints, manifests, audits, and summaries. Confirmation seeds remain unused.

## Stop conditions

Training/evaluation stops without bypass if validation fails, no lineage passes structural gates, no candidate passes all five global development trials, the action audit fails to execute, the long-horizon evaluation fails, a forbidden seed appears, or an artifact checksum differs. A failed gate is a result; thresholds are not weakened after observing the fresh lineages.

## Claim boundary after the next run

Even if every development gate passes, the result is still **development evidence**. A final superiority or lifetime claim requires a frozen candidate, untouched confirmation seeds, sufficient observed lifetime events or a prespecified survival analysis, and paper text that preserves the cross-study non-comparability boundary.
