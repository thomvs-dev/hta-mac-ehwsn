# Step 3 Local-CPU Bounded Probe Result Analysis

**Completed:** 10 August 2026  
**Analyzed:** 11 August 2026  
**Run:** `step3_v3_bounded_100ep_seed5599_cpu18`  
**Scope:** one-seed development probe; not model selection or publication evidence

## Decision

The frozen episode-100 checkpoint gate **failed**. Full training and held-out
Phase 3 evaluation are therefore **not authorized** from this checkpoint.

The run itself completed correctly: 100 episodes, 116,355 environment steps,
all 20 curriculum pairs visited, finite learning rows, no always-sleep collapse,
active but non-dominating CH-risk reward, and an intact episode-100 checkpoint.
The sole checkpoint-gate failure was greedy target QoS.

## Frozen gate result

| Gate | Result |
|---|---:|
| Exactly 100 episodes | Pass |
| Finite learning rows | Pass |
| No always-sleep collapse | Pass |
| CH-risk term active | Pass |
| CH-risk term non-dominating | Pass |
| Greedy target QoS | **Fail** |

The tail CH-risk contribution was 1.718% of absolute reward, below the 20%
dominance ceiling. The delivery-penalty geometry also passed its frozen 2-10%
active range. The result therefore cannot reasonably be explained as reward
collapse or an excessive lifetime penalty.

## Policy-stability trajectory

| Episode | Mean FND-free rounds | Throughput | Queue fairness | Target packets | Packets/J |
|---:|---:|---:|---:|---:|---:|
| 25 | 1179.90 | 97,512.05 | 0.719825 | 7,986.15 | 2,060.37 |
| 50 | 1180.55 | 97,457.15 | 0.717986 | 7,883.50 | 2,060.09 |
| 75 | 1173.95 | 97,646.85 | 0.739292 | 8,475.30 | 2,061.71 |
| 100 | 1170.75 | 97,863.65 | 0.745864 | 8,778.55 | 2,062.53 |

From episode 25 to 100, target packets increased by 792.40, throughput by
351.60, fairness by 0.02604, and efficiency by 2.16 packets/J. At the same
time, FND-free lifetime decreased by 9.15 rounds. These are development
observations from one seed, not statistically supported improvements.

## Exact QoS failure

At episode 100:

- joint QoS passes: **0/20**, required at least 18/20;
- delivery passes: **0/20**;
- stale-drop passes: **20/20**;
- fairness passes: **18/20**;
- delivery ratio: min 0.478295, mean 0.486326, max 0.500099;
- required delivery ratio: 0.55;
- stale ratio mean: 0.000541;
- fairness mean: 0.827900.

Delivery is therefore the universal blocking constraint. Even the best pair
missed the floor by 0.049901. Training longer is not justified: all four
stability snapshots had 0/20 joint QoS passes, so the bounded evidence does
not show convergence toward feasibility.

## CPU execution evidence

The launcher exposed 18 PyTorch intra-op threads and four inter-op threads on
20 logical processors. During the recorded monitoring interval (approximately
episodes 32-100), useful trainer-process utilization averaged 73.448% of total
logical-CPU capacity and peaked at 89.335%. No dummy CPU workload was used.
The early portion of the run was not included in that final monitoring mean,
so 73.448% must not be described as the whole-run average.

## Required next step

Do **not** extend this checkpoint to 500 episodes and do **not** evaluate it on
held-out seeds. First run a no-learning, development-only diagnostic on all 20
curriculum pairs that decomposes, per round:

1. target offered demand and feasible backlog;
2. requested, projected, and executed target allocation;
3. unused global budget and contention;
4. losses caused by target sleeping, CH forwarding capacity, target death,
   and schedule transitions;
5. achievable delivery ratio for an oracle budget-filling MAC policy under
   the same frozen CH schedule.

Proceed to another bounded candidate only if that diagnostic proves the 0.55
delivery floor is structurally reachable without modifying the CH schedule.
The next candidate should target the identified allocation bottleneck while
retaining CH-role risk conditioning, trajectory-order loss, concavity loss,
and the existing stop gate. If the oracle cannot reach 0.55, repair or
re-register the metric/environment contract before any further training.

## Evidence files

- `outputs/phase2/step3_v3_bounded_100ep_seed5599_cpu18/summary.json`
- `outputs/local_cpu_export/step3_v3_bounded_100ep_seed5599_cpu18/STEP3_BOUNDED_CHECKPOINT_GATE.json`
- `outputs/local_cpu_export/step3_v3_bounded_100ep_seed5599_cpu18/cpu_utilization.json`
- `outputs/local_cpu_export/step3_v3_bounded_100ep_seed5599_cpu18/stability_episode_100.pt`
- `outputs/local_cpu_export/step3_v3_bounded_100ep_seed5599_cpu18/episodes.jsonl`

