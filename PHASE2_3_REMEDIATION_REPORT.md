# HTA-MAC Phase 2 Remediation and Phase 3 Pilot Report

Date: 2026-07-29

## Executive status

Phase 2 now has an authoritative **PASS** checkpoint trained against frozen
per-round HEART-CH schedule replay. The five-seed Phase 3 held-out pilot also
completed without crashes or non-finite metrics. HTA-MAC had no observed first
node death (FND) or half-node death (HND) before frozen-schedule exhaustion in
any held-out seed, so lifetime is reported as right-censored rather than as an
invented numeric death time.

This is a strong pilot result, not a publication-level superiority claim. With
only five pairs, the smallest attainable two-sided Wilcoxon p-value is 0.0625.
The policy also trades queue service quality for survival: its stale-packet drop
count is much higher and its Jain fairness is lower than the comparison
policies. Phase 4 still needs 30 paired trials.

## Scope preserved

- HEART-CH CH selection remains frozen and exogenous.
- No CH-policy retraining was performed.
- No routing model or Pointer Network was introduced.
- All evaluation policies use the same seed-specific frozen schedule,
  environment, HMM assets, radio model, queue model, and idle-energy accounting.
- The environment frame capacity remains `T=24`; HTA-MAC voluntarily enforces an
  internal learned-action projection cap of 8, which satisfies `sum(a_i) <= T`.

## Defects and mismatches found

### Queue-infeasible learned actions

The initial checkpoint could allocate more slots than a node had queued packets.
Queue-derived per-node caps are now enforced in random exploration, greedy
action selection, replay transitions, and Bellman target projection. The
inference policy also uses the active environment budget rather than blindly
trusting a checkpoint budget.

### Missing target-CH death reward

The fixed-cluster reward counted member deaths but not death of the target CH,
even though CH death ended the episode. The death term now counts a newly dead
CH exactly once. The target-cluster terminal rule was also made explicit and
tested.

### Fixed-cluster training did not match evaluation

The first curriculum froze CH identity and membership for an episode. It passed
its internal gate but failed held-out evaluation because Phase 3 changes CH
roles and cluster membership every round. A new scheduled training wrapper now
uses the exact frozen per-round replay mechanism used in evaluation.

Global node identity is preserved across transitions: the agent receives 100
node branches, masks only the current target-cluster members, and projects the
budget over that mask. This avoids pairing one node's current state with an
unrelated next-round member merely because both occupied the same compact array
index.

### Dynamic-schedule edge cases

Two dynamic-only correctness cases were caught during smoke/full runs:

1. a temporarily empty cluster caused by reassignment was initially treated as
   if all members had died; it now continues unless a real death, truncation, or
   dead scheduled CH occurs;
2. frozen replay can later nominate a CH already dead under the MAC-controlled
   energy trajectory; all target branches are now masked and forced to zero for
   that frame, matching the evaluation-policy contract.

The partial runs exposing these defects were stopped and are not authoritative.

### Reward and allocation diagnosis

The uncapped dynamic checkpoint passed Phase 2 but failed independent validation:
median FND was 107 rounds versus 129 static and 190 S2A2MAC-adapted. Its reward
balance was packet-heavy (55.7% packet contribution, 17.7% idle, 0.31% death),
and it allocated about 79 slots per network round on a traced seed, versus 72.6
static and 48.0 S2A2MAC-adapted.

A weight-3 idle / weight-10 death fine-tune improved throughput and FND slightly
but remained inadequate at projection budget 24. An idle-weight-10 run was
stopped because returns saturated toward the C51 lower support while allocation
density did not change; continuing would have violated the broken-reward-signal
stop rule.

A development-only projection sweep over `{8,12,16,20,24}` showed a monotonic
lifetime/efficiency improvement as the internal cap decreased. Budget 8 was then
activated during a fresh 500-episode authoritative training run rather than
used only as a post-hoc inference override.

## Authoritative Phase 2 training

Command:

```powershell
python -B experiments\train_phase2_dynamic_curriculum.py `
  --episodes 500 --max-steps 150 --learn-every 4 `
  --idle-weight 3.0 --death-weight 10.0 `
  --epsilon-start 0.2 --epsilon-end 0.05 `
  --projection-budget 8 `
  --initial-checkpoint outputs\phase2\dynamic_finetune_idle3_death10_200ep\branching_c51.pt `
  --run-name authoritative_dynamic_budget8_500ep
```

Printed gate evidence:

```text
CURRICULUM_PAIRS=25
MAX_PADDED_BRANCHES=100
EPISODES_COMPLETED=500
FULL_CURRICULUM_SEEN=True
ALWAYS_SLEEP_COLLAPSE=False
REWARD_PATHOLOGICAL_DOMINATION=False
GREEDY_MEAN_PACKETS=827.3200
S8_S1_Q_MAX_ABS_DIFF=0.02627563
CONVERGENCE_PASS=True
PHASE2_CURRICULUM_GATE_PASS=True
```

The final two 50-episode reward means were 145.3446 and 132.4639, an 8.862%
relative change under the predefined 10% convergence threshold. No reward term
crossed the 80% domination threshold. The last-50 contribution fractions were:

| Reward term | Fraction |
|---|---:|
| packets delivered | 0.5692 |
| queue fairness | 0.1737 |
| idle energy | 0.0968 |
| declining allocation | 0.0833 |
| high-harvest alignment | 0.0771 |
| deaths | 0.0000 |

The zero death contribution means no target death occurred in the last 50
training episodes; it is not evidence that the death weight is independently
validated.

Checkpoint SHA-256:
`0EF29EFAFF04EC1CB652C84A432A53BD0C41D7C68DC9DECFCADF9C277247C2FF`.

## Independent development validation

Seeds 2400-2404 were distinct from training seeds 2300-2304 and held-out seeds
3100-3104. Across all five development-validation schedules, HTA-MAC had zero
observed FND and zero observed HND before schedule exhaustion (1,635-1,685
rounds). Median throughput was 21,403 packets and median idle energy 17.69 J.
This validation was used to approve the authoritative checkpoint for the final
held-out pilot.

## Held-out Phase 3 pilot

Command:

```powershell
python -B experiments\run_phase3_pilot.py `
  --seeds 3100,3101,3102,3103,3104 `
  --horizon 3000 `
  --run-name heldout_pilot_authoritative_budget8_censor_aware `
  --skip-compatibility
```

Structural evidence:

```text
PRIMARY_RUNS=35/35
FAILURES=0
ALL_METRICS_FINITE=True
PHASE3_STRUCTURAL_GATE_PASS=True
```

Selected median +/- IQR results:

| Policy | FND/HND status | Throughput | Idle J | Jain fairness | Packets/J | Stale drops |
|---|---|---:|---:|---:|---:|---:|
| Static equal | observed, KM medians 132/163 | 13,247 +/- 256 | 45.137 +/- 0.511 | 0.9221 +/- 0.0276 | 240.53 +/- 3.07 | 17,973 +/- 1,697 |
| S2A2MAC-adapted | observed, KM medians 201/354 | 22,637 +/- 481 | 46.032 +/- 0.222 | 0.8403 +/- 0.0087 | 381.51 +/- 5.80 | 39,389 +/- 971 |
| HTA-MAC | 0/5 FND events, 0/5 HND events | 21,374 +/- 437 | 17.057 +/- 1.167 | 0.7804 +/- 0.0215 | 592.26 +/- 12.31 | 143,335 +/- 1,901 |

HTA-MAC therefore delivered 1,263 fewer median packets than S2A2MAC-adapted,
but used about 63% less idle energy and achieved about 55% higher packets/J.
These percentages are descriptive pilot quantities, not inferential claims.
HTA-MAC's substantially worse stale-drop count and lower fairness must be
reported prominently rather than hidden behind lifetime metrics.

## Censor-aware lifetime analysis

The common restriction horizon is the minimum schedule coverage across paired
runs: 1,633 rounds. For each endpoint, an observed death uses its event round;
an unobserved death uses its censor round and is restricted at 1,633. Missing
values are never replaced with infinity or the requested 3,000-round horizon.

| Endpoint/policy | Events | Censored | KM median | Restricted mean event-free rounds |
|---|---:|---:|---:|---:|
| HTA FND | 0 | 5 | not reached | 1,633.0 |
| S2A2MAC FND | 5 | 0 | 201 | 200.0 |
| Static FND | 5 | 0 | 132 | 132.0 |
| HTA HND | 0 | 5 | not reached | 1,633.0 |
| S2A2MAC HND | 5 | 0 | 354 | 346.4 |
| Static HND | 5 | 0 | 163 | 162.2 |

Paired HTA-minus-S2A2MAC restricted-time differences are +1,432 rounds for FND
and +1,279 rounds for HND. Both exact/automatic two-sided Wilcoxon tests return
`p=0.0625` for five pairs. This is not significant at 0.05 and demonstrates why
Phase 4 needs 30 trials.

Raw-trial CSV SHA-256:
`7529E8CCB99F35F6E0BCC6552B62BD9C5B8DE95B2E584548E17C51FD6D076438`.

## Decisions now locked

- Use scheduled, identity-preserving dynamic curriculum, not fixed-cluster-only
  training, for the authoritative checkpoint.
- Keep the environment capacity at `T=24` and record the HTA policy's internal
  projection cap as 8; do not describe the environment itself as `T=8`.
- Use idle weight 3 and death weight 10 for this checkpoint; report the weight
  search as development tuning.
- Treat FND and HND as right-censored when no event occurs before schedule
  exhaustion.
- Use Kaplan-Meier status plus common-horizon restricted event-free time;
  retain paired Wilcoxon only on the restricted paired endpoint.
- Do not use the original novelty sentence claiming S2A2MAC cannot
  differentiate nodes. The source-based adapted baseline already uses per-node
  active layers.

## Remaining work and current uncertainty

1. Phase 4: run 30 paired seeds for all seven policies and archive raw CSVs.
2. Confirm whether the very high stale-drop rate is acceptable for the target
   application; if not, tune a service/QoS constraint on development seeds and
   retrain before Phase 4. This choice materially changes the paper's claim.
3. Run the planned `n_max`, reward, hybrid-source, and idle-energy ablations only
   after the main 30-trial checkpoint is frozen.
4. The thermal HMM remains an auxiliary default model, not a dataset-trained
   real thermal predictor.
5. Schedule exhaustion is driven by the frozen upstream HEART-CH replay. It is
   a hard observation boundary and must not be bypassed by repeating stale
   frames.

## Evidence index

- Authoritative training summary:
  `outputs/phase2/authoritative_dynamic_budget8_500ep/summary.json`
- Authoritative episode log:
  `outputs/phase2/authoritative_dynamic_budget8_500ep/episodes.jsonl`
- Authoritative checkpoint:
  `outputs/phase2/authoritative_dynamic_budget8_500ep/branching_c51.pt`
- Development budget sweep:
  `outputs/phase2/development_budget_sweep_tuned_idle3.json`
- Development validation:
  `outputs/phase3/development_validation_authoritative_budget8/summary.json`
- Held-out raw trials:
  `outputs/phase3/heldout_pilot_authoritative_budget8_censor_aware/raw_trials.csv`
- Held-out censor-aware report:
  `outputs/phase3/heldout_pilot_authoritative_budget8_censor_aware/summary.json`
- Baseline mechanism provenance: `BASELINE_PROVENANCE.md`

Final regression command:

```powershell
python -B -m pytest validation -q -p no:cacheprovider
```

Result: **25 passed**. The 49 warnings are upstream library deprecations; no test
failed.