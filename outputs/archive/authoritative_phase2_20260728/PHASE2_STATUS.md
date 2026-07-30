# HTA-MAC authoritative Phase 2 status

Status: **PASS for the fixed-cluster training gate** on 2026-07-28.

This status means the Branching Dueling C51 agent trained without numerical or
always-sleep failure on the required single fixed cluster. It does **not** claim
multi-trial or publication-level superiority; Phase 3 baselines and Phase 4
paired trials remain unexecuted.

## Authoritative training command

```powershell
python -B experiments\train_phase2_fixed_cluster.py `
  --episodes 500 --max-steps 150 `
  --run-name authoritative_500ep_seed2100
```

Printed gate evidence:

```text
EPISODES_COMPLETED=500
ALWAYS_SLEEP_COLLAPSE=False
REWARD_PATHOLOGICAL_DOMINATION=False
GREEDY_MEAN_PACKETS=1006.0000
S8_S1_Q_MAX_ABS_DIFF=0.61809874
CONVERGENCE_PASS=True
PHASE2_GATE_PASS=True
```

The run archived 35,137 transitions. The previous-50 and last-50 mean rewards
were 83.7032 and 86.3369, respectively, a 3.1465% relative change under the
predefined 10% stability threshold. Greedy evaluation was deterministic across
10 resets and never selected an all-zero cluster action.

## Reward-term inspection

Absolute contribution fractions over the last 50 episodes were:

| Term | Fraction |
|---|---:|
| packets delivered | 0.4456 |
| idle energy | 0.2296 |
| queue fairness | 0.1392 |
| high-harvest alignment | 0.0997 |
| declining-state allocation penalty | 0.0859 |
| member deaths | 0.0000 |

No active term crossed the predefined pathological-domination threshold of
0.80. The death term was logged but inactive on this fixed-cluster horizon;
that is a limitation, not evidence that its weight is validated.

At equal normalized residual energy (0.5), the trained branch returned a
maximum absolute S8-versus-S1 Q-value difference of 0.61810. Removing the common
state-value offset, the maximum action-marginal difference was 0.04532. Both
synthetic nodes still preferred action zero when evaluated in isolation, so the
result proves Q differentiation but does not by itself prove preferential S8
allocation.

## Fixed-cluster control inspection

Command:

```powershell
python -B experiments\evaluate_phase2_fixed_cluster.py
```

One deterministic seed-2100 episode produced:

| Policy | Reward | Packets | Episode steps | Mean slots/step |
|---|---:|---:|---:|---:|
| always sleep | 0.0000 | 0 | 150 | 0.000 |
| static equal | 89.4077 | 1240 | 62 | 20.000 |
| random budgeted | 93.9330 | 1221 | 64 | 24.000 |
| trained greedy | 92.4555 | 1006 | 75 | 23.747 |

Relative to static equal TDMA, the learned policy gained 3.0477 reward units,
extended this fixed episode by 13 steps, and reduced idle-energy consumption by
0.9144 J, but delivered 234 fewer packets. It did not exceed the random-budgeted
control's reward. Therefore superiority is explicitly **not established**.
These are diagnostic single-episode controls, not Phase 4 results.

## Architecture and scope

- Input per member: 18 revised MAC features plus the frozen 32-dimensional
  upstream ST-GCN embedding (50 total).
- Action per member: 0-3 slots; greedy marginal-Q budget projection enforces
  `sum(a_i) <= 24`.
- Shared node branch: dueling distributional C51 head with 51 atoms.
- Reused infrastructure: prioritized replay, target network, and distributional
  update adapted from HEART-CH's Rainbow design.
- Training cluster: immutable seed-2100 HEART-CH snapshot, target cluster 2,
  CH 30, 20 members. This fixed snapshot is required only for the Phase 2
  convergence gate. Full comparisons must use shared exogenous per-round frozen
  schedule replay.

## Evidence

- Training summary: `outputs/phase2/authoritative_500ep_seed2100/summary.json`
- Raw episode log: `outputs/phase2/authoritative_500ep_seed2100/episodes.jsonl`
- Checkpoint: `outputs/phase2/authoritative_500ep_seed2100/branching_c51.pt`
- Control report: `outputs/logs/phase2_fixed_cluster_controls.json`
- Reward calibration: `outputs/logs/phase2_reward_calibration.json`
- Smoke run: `outputs/phase2/smoke_30ep_seed2100/`

Validation after training: `12 passed` using
`python -B -m pytest validation -q -p no:cacheprovider`.
