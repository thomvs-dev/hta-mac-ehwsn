# HTA-MAC Comprehensive Progress, Decisions, and Open-Issues Report

**Project:** HMM-Trajectory-Aware Medium Access Control (HTA-MAC)  
**Repository:** `F:\WSN\matlab\stage2\hta-mac`  
**Report date:** 2026-07-29  
**Current verified boundary:** Phase 2 passed; Phase 3 held-out pilot completed;
Phase 4 has not started.

---

## 1. Purpose and evidence standard

This report is the consolidated technical handoff for the instructor and future
research sessions. It records:

1. what was inspected, implemented, trained, and evaluated;
2. which results came from real archived runs;
3. the engineering rationale for the important decisions;
4. defects, failed experiments, and rejected approaches;
5. the current scientific interpretation of the results;
6. unresolved doubts and publication risks;
7. the exact work that remains.

This report supersedes `HTA_MAC_PROGRESS_DECISION_REPORT.md` as the current
project-wide status document. The older report remains useful as a
pre-Phase-2 historical snapshot. The phase-specific status files and archived
JSON/CSV outputs remain the primary evidence.

“Reasoning” in this document means inspectable engineering rationale tied to
code, measurements, or experimental controls. It does not attempt to reproduce
private chain-of-thought, and it does not treat an undocumented idea as
scientific evidence.

No result in this report is invented. Pilot results are identified as pilot
results, censored lifetime outcomes remain censored, and unsuccessful training
runs are not presented as positive evidence.

---

## 2. Executive summary

The project has progressed from an unverified proposal to a functioning,
tested HTA-MAC research simulator with:

- frozen HEART-CH cluster-head schedule replay;
- explicit idle-listening energy accounting;
- an 18-dimensional per-node MAC state;
- hybrid solar and thermal HMM sampling;
- rectified harvest moments;
- bounded packet queues with three-round TTL expiry;
- a marginal-Q budget-projection layer;
- a Branching Dueling C51 DQN with queue-feasible actions and targets;
- seven policies evaluated through one interface in the same environment;
- an authoritative Phase 2 checkpoint;
- a completed five-seed held-out Phase 3 pilot;
- censor-aware lifetime reporting;
- 25 passing validation tests.

The main scientific result so far is a strong but incomplete tradeoff:

- the authoritative budget-8 HTA-MAC policy greatly reduces idle energy and
  improves energy efficiency;
- no FND or HND event was observed in any of the five held-out schedule
  windows;
- however, it delivers fewer packets than S2A2MAC-adapted, has lower Jain
  fairness, and drops far more stale packets.

Therefore, the project is **not ready for a publication-level superiority
claim or the final 30-trial Phase 4 campaign** until the intended
lifetime-versus-QoS operating point is selected. The existing budget-8
checkpoint is defensible as a lifetime-optimized reference. A balanced
budget-12 candidate should be trained and assessed only on development seeds
before deciding which checkpoint becomes the primary Phase 4 policy.

---

## 3. Phase-by-phase status

| Phase | Status | Verified outcome |
|---|---|---|
| Phase 0: Foundation | PASS under corrected empirical foundation | Frozen released HEART-CH artifact reproduced `T_FND=1100.6 +/- 44.18`, not the manuscript’s `1191.3 +/- 40.0` |
| Phase 1: Structural frame | PASS | Idle mechanism, energy conservation, HMM distribution checks, queue semantics, budget projection, and frozen schedule replay validated |
| Phase 2: Branching DQN | PASS | Authoritative 500-episode dynamic curriculum checkpoint passed convergence, collapse, reward-balance, and Q-differentiation gates |
| Phase 3: Baselines/pilot | Pilot complete | 35/35 held-out runs completed; censor-aware analysis added; QoS tradeoff remains unresolved |
| Phase 4: 30-trial evaluation | NOT STARTED | Must wait for primary operating-point decision |
| Phase 5: Ablations/bound | NOT STARTED | Planned only after main checkpoint is frozen |
| Phase 6: Manuscript/handover | PARTIAL PREPARATION ONLY | Provenance and limitations text exist; figures, final tables, DOI audit, and manuscript remain |

---

## 4. Scope that was deliberately preserved

The implementation has remained inside the bounded contribution specified for
HTA-MAC:

- HEART-CH cluster-head selection is frozen and is not retrained.
- HTA-MAC controls only intra-cluster TDMA slot allocation.
- No Pointer Network or routing learner was added.
- HERMES results were not merged with HTA-MAC results.
- All comparison policies use the same seed-specific schedule, topology,
  mobility, HMM assets, queue rules, radio model, and idle-energy accounting.
- The upstream `W=10` setting remains unchanged for this paper.
- The thermal component is explicitly treated as a synthetic auxiliary, not as
  a dataset-trained thermal predictor.

### Why this boundary was retained

The previous HERMES direction combined incompatible simulators, environment
settings, and learning tasks. Reintroducing online CH retraining or learned
routing would make it difficult to attribute a measured improvement to the MAC
policy. Shared frozen schedule replay creates a controlled MAC-layer
comparison: every policy receives the same exogenous clustering decisions.

---

## 5. Phase 0: foundation and provenance

### 5.1 Upstream assets inspected and frozen

The clean upstream implementation used for HEART-CH assets is:

```text
F:\WSN\matlab\stage2\final_repo
```

Verified upstream commit:

```text
d96abce25237feb2b6d6c660f6b4d605feb94330
```

Frozen HEART-CH checkpoint:

```text
outputs/checkpoints/model_v91_throughput.pt
SHA-256:
ccb572901e263a50954c9a9b0746cf596193d0018502e0c7b26b623e6d287c5f
```

Solar Stage 1 HMM:

```text
outputs/stage1_params.mat
SHA-256:
8a864e9e10235037fb86b71fc6cc3a35c9a8637593c965c797318eaeff52252c
```

The solar artifact is a trained eight-state HMM. No trained four-state thermal
artifact was found. The upstream default thermal parameters were therefore
frozen as:

```text
core/hmm/thermal_auxiliary_params.npz
SHA-256:
3d22e56e47a499884b50f88d2598124153ab99386ad62c327835d50cbc46c845
provenance: synthetic_auxiliary_from_heart_ch_defaults
trained: false
```

### 5.2 HEART-CH reproduction result

A fresh 30-trial evaluation using seeds 1000-1029 produced:

```text
T_FND mean             = 1100.6 rounds
population std         = 44.18189674516023 rounds
median                 = 1100.0 rounds
IQR                    = 58.25 rounds
NaN/crash trials       = 0
```

The original manuscript number `1191.3 +/- 40.0` was not reproduced. Checks
using the manuscript seed range and checkpoint metadata also failed to recover
that value.

### 5.3 Decision and rationale

The executable baseline is the independently reproduced
`1100.6 +/- 44.18`. The manuscript value is retained only as a reported
historical result.

This decision prevents an unavailable or under-specified number from being used
as the control for HTA-MAC. Paired comparisons must use measurements produced
inside the same implementation and evaluation protocol.

### 5.4 Phase 0 outputs

- `config/phase0_acceptance.yaml`
- `outputs/logs/phase0_corrected_gate.json`
- `validation/phase0_gate_corrected.py`
- `PHASE0_ORIGINAL_GATE_STATUS.md`
- `PHASE0_STATUS.md`
- `HERMES_ARTIFACT_AUDIT.md`

---

## 6. HERMES audit and reuse decisions

The `hermes/` tree was inspected rather than copied wholesale. It contained
useful diagnostic patterns but no compatible substitute for the released
HEART-CH checkpoint or a trained thermal HMM.

Rejected artifacts included:

- an analytical HEART comparator instead of the neural checkpoint;
- a checkpoint using `W=20`, `F=33`, and quorum-cover selection;
- a simulator with a 200 m range and reflective mobility;
- synthetic thermal transitions generated from an assumed persistence value;
- learned evaluations explicitly marked as excluding HERMES.

### Reason for rejection

Those artifacts solve a different problem under different assumptions.
Importing their metrics or checkpoints would recreate the disconnected-system
failure that the HTA-MAC project was created to avoid.

Useful mechanisms were reimplemented or adapted only when their semantics could
be validated inside the unified HTA-MAC environment.

---

## 7. Phase 1: structural environment

### 7.1 Idle-listening energy

Implemented in:

```text
core/energy/idle_model.py
```

The implementation uses:

```text
E_idle = idle_slots * E_elec * slot_bit_times
```

This interprets `E_elec` as electronics energy per bit-time. It avoids
introducing an unsupported hardware power constant.

Two defensible slot-duration interpretations and an off case were retained:

| Idle interpretation | Slot bit-times | Median T_FND, five gate seeds |
|---|---:|---:|
| Full data-slot listening | 4000 | 127 |
| Header/control-only listening | 100 | 818 |
| Idle disabled | 0 | 920 |

The expected ordering passed. The large full-slot effect is therefore an
active modeled mechanism, not a hidden arithmetic error. It remains a
sensitivity result and must not be presented as universal hardware behavior.

### 7.2 Energy and HMM validation

The authoritative Phase 1 gate printed:

```text
FROZEN_CH_COUNT=5
ENERGY_TRACE_MAX_ERROR_J=0.000e+00
DETERMINISTIC=True
BUDGET_VIOLATIONS=0/1000
HMM_KS_SOLAR_D=0.0116,P=0.511606
HMM_KS_THERMAL_D=0.0157,P=0.169934
T_FND_IDLE_ON_OFF_MEDIAN=127.0/920.0,SHIFT=793.0
IDLE_SENSITIVITY_MEDIANS=127.0/818.0/920.0
IDLE_SENSITIVITY_ORDER_PASS=True
AUTHORITATIVE_PHASE_1_GATE=PASS
```

The 20-round hand trace matched the bounded energy equation with zero recorded
absolute error. The KS tests show consistency with the inherited parameterized
emission distributions; they do not establish agreement with real-world solar
or thermal traces.

### 7.3 State and forecast corrections

The per-node observation has 18 dimensions:

- normalized residual energy;
- next-round harvest forecast mean and variance;
- eight solar state-conditioned transition probabilities;
- four thermal state-conditioned transition probabilities;
- normalized queue occupancy;
- previous allocation;
- normalized cluster size.

The probability blocks are now called **state-conditioned transition
probabilities**, not Bayesian posteriors. No forward-filter observation update
was found upstream.

Forecast mean and variance use the rectified Gaussian moments from HEART-CH
Eqs. 13-14 rather than an unrectified scaled-Gaussian approximation.

### 7.4 Queue semantics

The queue model was fixed before training:

- one packet is generated per alive node per round;
- one allocated slot transmits at most one 4000-bit packet;
- packets expire after three rounds;
- expired packets are counted as `dropped_stale_packets`;
- static-pilot maximum backlog was four packets;
- `q_max=5` was selected from that measurement;
- no overflow was observed in the held-out pilot.

This makes allocation levels above one useful while preventing an unbounded
backlog from obscuring service fairness.

### 7.5 Frame capacity and action range

Measured median cluster size was 18 members. The shared frame capacity was set
to:

```text
T = ceil(1.3 * 18) = 24 slots
n_max = 3 slots per node
```

The earlier values `T=40` and `T=100` were gate placeholders, not research
decisions. `T=24` introduces real contention while retaining a modest margin
above the median cluster size.

### 7.6 Frozen schedule replay

The first implementation froze one CH/member snapshot for the whole episode.
The fixed CHs became the first deaths in both idle-on and idle-off arms, masking
the member idle effect.

It was replaced with per-round frozen HEART-CH schedule replay:

```text
core/ch_selection/frozen_schedule.py
core/ch_selection/frozen_schedule_full.py
envs/scheduled_mac_env.py
```

The same schedule is replayed for paired policies. Schedules request a
3000-round horizon, stop when the frozen upstream policy can no longer select a
CH, and never repeat the last frame. Endpoints not reached before that boundary
are right-censored.

### 7.7 Why shared replay was selected

Re-evaluating the frozen CH network online using each MAC policy’s changed
energy trajectory would allow the upstream CH decisions to diverge. That is a
valid future co-adaptation study, but it would confound the present MAC-layer
comparison.

Shared exogenous replay was selected for causal attribution. The limitation is
that later CH decisions and ST-GCN embeddings do not respond to the energy
trajectory created by HTA-MAC. This must be stated explicitly in the paper.

---

## 8. Phase 2: Branching DQN implementation

### 8.1 Agent architecture

The Phase 2 agent is implemented in:

```text
agents/branching_dqn.py
agents/prioritized_replay.py
agents/reward_model.py
agents/budget_projection.py
```

It uses:

- a shared input trunk;
- per-node dueling value/advantage branches;
- C51 distributional Q-learning;
- prioritized replay;
- a target network;
- four discrete actions per node: `0,1,2,3`;
- marginal-Q knapsack-style projection;
- masking for nodes outside the active target cluster;
- queue-feasible action caps;
- queue-feasible Bellman-target caps.

The model uses global node identity across schedule transitions. A fixed
100-branch representation prevents one physical node’s current state from
being paired with an unrelated node’s next state merely because cluster
membership changed.

### 8.2 Reward calibration

Reward components were logged separately and normalized using observed natural
scales. The authoritative checkpoint uses:

```text
packet weight      = 2.0
idle weight        = 3.0
death weight       = 10.0
high-harvest       = 0.5
declining-state    = 0.5
queue fairness     = 0.5
```

The final 50-episode absolute-contribution fractions were:

| Reward component | Fraction |
|---|---:|
| Packets delivered | 0.5692 |
| Queue fairness | 0.1737 |
| Idle energy | 0.0968 |
| Declining-state allocation | 0.0833 |
| High-harvest alignment | 0.0771 |
| Deaths | 0.0000 |

No term crossed the predefined 80% pathological-domination threshold. The zero
death fraction means no target death occurred in the last 50 training episodes;
it does not independently validate the death weight.

### 8.3 Correctness defects found and fixed

#### Queue-infeasible actions

The first agent could request more slots than a node had queued packets.
Queue-derived per-node caps are now enforced during exploration, greedy
selection, replay storage, and target projection.

#### Missing target-CH death penalty

The fixed-cluster reward counted member deaths but not the target CH death even
though that event terminated the episode. A newly dead target CH is now counted
exactly once.

#### Training/evaluation mismatch

The original curriculum kept CH identity and membership fixed. It passed its
own gate but failed evaluation where clusters changed every round. It was
replaced by `envs/dynamic_cluster_training_env.py`, which uses the same frozen
schedule mechanism as Phase 3.

#### Empty-cluster interpretation

A temporarily empty cluster caused by reassignment was initially treated as a
death event. It now continues unless a real death, dead scheduled CH,
truncation, or schedule boundary occurs.

#### Dead scheduled CH

The frozen replay can select a CH that is dead under a MAC-controlled energy
trajectory. All target branches are now masked and forced to zero for that
frame.

### 8.4 Training experiments and what they showed

#### Historical fixed-cluster checkpoint

It passed a local sanity gate but generalized poorly because it did not see
round-to-round CH/member changes. It is retained only as historical evidence.

#### Static multi-seed curriculum

`outputs/phase2/curriculum_600ep_dev10` passed its internal gate but failed
held-out generalization. The conclusion was that seed diversity alone cannot
repair a structural mismatch between fixed membership during training and
dynamic membership during evaluation.

#### Dynamic schedule-matched checkpoint at budget 24

`outputs/phase2/dynamic_curriculum_500ep_dev5_h150_v2` passed its training gate,
but development median FND was approximately 107 rounds. This was worse than
the static and S2A2MAC-adapted development references.

Diagnostics showed that the policy was packet-heavy and allocated roughly 79
slots per network round, compared with about 72.6 for static and 48 for
S2A2MAC-adapted. Learning had succeeded numerically but not at the desired
energy/service operating point.

#### Idle/death-weight fine-tuning

A weight-3 idle, weight-10 death fine-tune improved FND and throughput modestly
but remained inadequate at budget 24.

An idle-weight-10 run was stopped after 84 episodes because C51 returns moved
toward the lower support while allocation density did not decrease. Continuing
would have violated the requirement to stop on a broken reward signal.

#### Development budget sweep

Using the weight-3/death-10 tuned checkpoint, the inference projection cap was
swept over `{8,12,16,20,24}` on development seeds 2400-2404:

| Internal cap | Median FND | Median HND | Throughput | Fairness | Stale drops | Idle J | Packets/J |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 508 | 915 | 35,059 | 0.9793 | 58,624 | 41.34 | 535.49 |
| 12 | 263 | 568 | 28,704 | 0.9455 | 35,491 | 41.86 | 473.57 |
| 16 | 178 | 349 | 24,222 | 0.9071 | 28,189 | 43.12 | 412.57 |
| 20 | 140 | 224 | 20,635 | 0.8585 | 22,688 | 43.65 | 359.00 |
| 24 | 112 | 161 | 17,310 | 0.8160 | 18,168 | 44.05 | 310.39 |

These are development results from a non-authoritative 200-episode fine-tuned
checkpoint. They demonstrate the operating tradeoff but are not a substitute
for training and validating an authoritative candidate at the selected cap.

### 8.5 Authoritative Phase 2 checkpoint

The budget-8 operating point was activated during a fresh 500-episode
schedule-matched curriculum run, rather than being used only as a post-hoc
inference override.

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

Training covered five development seeds and all five cluster ranks:

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

The last two 50-episode reward means were 145.3446 and 132.4639, an 8.862%
relative change under the predefined 10% convergence threshold.

Authoritative checkpoint:

```text
outputs/phase2/authoritative_dynamic_budget8_500ep/branching_c51.pt
SHA-256:
0EF29EFAFF04EC1CB652C84A432A53BD0C41D7C68DC9DECFCADF9C277247C2FF
```

### 8.6 Phase 2 interpretation

The Phase 2 gate establishes that:

- the agent trained without numerical failure;
- the training curriculum was fully visited;
- the policy did not collapse to always sleeping;
- no logged reward component pathologically dominated;
- Q-values differentiated high- and low-harvest states;
- the selected convergence criterion passed.

It does **not** establish:

- publication-level superiority;
- general real-world thermal forecasting validity;
- optimality of budget 8;
- acceptable application-level delay or stale-drop behavior;
- statistical significance against baselines.

---

## 9. Phase 3: common-policy environment and held-out pilot

### 9.1 Policies

All policies implement the same `MACPolicyInterface` and run inside the same
scheduled MAC environment:

1. static equal TDMA;
2. energy proportional;
3. harvest proportional;
4. S2A2MAC-adapted;
5. FFSS-adapted;
6. HTA-MAC;
7. random-budgeted diagnostic.

The seventh policy is explicitly a diagnostic, not a literature baseline.

### 9.2 Baseline-provenance correction

The initial novelty sentence said that S2A2MAC structurally could not
differentiate nodes within a cluster. Source inspection showed that the adapted
mechanism uses per-node active layers. That sentence must not be used.

The defensible distinction is instead that HTA-MAC uses:

- learned per-node discrete allocation;
- hybrid solar and thermal trajectory information;
- explicit idle-listening energy accounting;
- a branching value architecture under a shared slot constraint.

All novelty wording still requires a final primary-source literature audit
before manuscript submission.

### 9.3 Independent development validation

Seeds 2400-2404 were distinct from training seeds 2300-2304 and held-out pilot
seeds 3100-3104.

The authoritative budget-8 model had zero observed FND and zero observed HND
events in all five development-validation schedule windows. Median throughput
was 21,403 packets and median idle energy was 17.69 J.

### 9.4 Held-out pilot execution

Command:

```powershell
python -B experiments\run_phase3_pilot.py `
  --seeds 3100,3101,3102,3103,3104 `
  --horizon 3000 `
  --run-name heldout_pilot_authoritative_budget8_censor_aware `
  --skip-compatibility
```

Structural result:

```text
PRIMARY_RUNS=35/35
FAILURES=0
ALL_METRICS_FINITE=True
PHASE3_STRUCTURAL_GATE_PASS=True
```

### 9.5 Selected held-out results

| Policy | Lifetime status | Throughput | Idle J | Fairness | Packets/J | Stale drops |
|---|---|---:|---:|---:|---:|---:|
| Static equal | KM FND/HND 132/163 | 13,247 +/- 256 | 45.137 +/- 0.511 | 0.9221 +/- 0.0276 | 240.53 +/- 3.07 | 17,973 +/- 1,697 |
| S2A2MAC-adapted | KM FND/HND 201/354 | 22,637 +/- 481 | 46.032 +/- 0.222 | 0.8403 +/- 0.0087 | 381.51 +/- 5.80 | 39,389 +/- 971 |
| HTA-MAC | 0/5 FND and 0/5 HND events | 21,374 +/- 437 | 17.057 +/- 1.167 | 0.7804 +/- 0.0215 | 592.26 +/- 12.31 | 143,335 +/- 1,901 |

Relative to S2A2MAC-adapted, HTA-MAC:

- delivered 1,263 fewer median packets;
- used approximately 63% less idle energy;
- achieved approximately 55% higher packets/J;
- had substantially lower fairness;
- produced more than three times as many stale drops.

These percentages are descriptive pilot quantities, not inferential claims.

### 9.6 Censor-aware lifetime handling

HTA-MAC had no observed FND or HND events before schedule exhaustion in any
held-out seed. Those outcomes are right-censored; they are not infinity and
must not be replaced by the requested 3000-round horizon.

The common restriction horizon is 1,633 rounds:

| Endpoint/policy | Events | Censored | KM median | Restricted mean event-free rounds |
|---|---:|---:|---:|---:|
| HTA FND | 0 | 5 | Not reached | 1,633.0 |
| S2A2MAC FND | 5 | 0 | 201 | 200.0 |
| Static FND | 5 | 0 | 132 | 132.0 |
| HTA HND | 0 | 5 | Not reached | 1,633.0 |
| S2A2MAC HND | 5 | 0 | 354 | 346.4 |
| Static HND | 5 | 0 | 163 | 162.2 |

Paired HTA-minus-S2A2MAC restricted-time differences are:

```text
FND-free time: +1432 rounds
HND-free time: +1279 rounds
```

Both five-pair two-sided Wilcoxon p-values are `0.0625`. The pilot therefore
does not establish significance at 0.05. With five nonzero pairs, 0.0625 is the
smallest attainable two-sided exact Wilcoxon p-value.

### 9.7 Honest Phase 3 conclusion

The budget-8 policy is a strong lifetime/idle-energy policy, but it partly
achieves survival by serving traffic less aggressively. The stale-drop and
fairness costs are not secondary presentation details; they determine what
claim the paper can honestly make.

The current evidence supports:

> HTA-MAC exposes a controllable lifetime-versus-service tradeoff and the
> budget-8 policy strongly favors lifetime and energy efficiency.

The current evidence does not yet support:

> HTA-MAC is unconditionally superior to all baselines.

---

## 10. Decision register

| Decision | Reason | Evidence/impact | Status |
|---|---|---|---|
| Use reproduced HEART-CH value `1100.6 +/- 44.18` | Original `1191.3` could not be reproduced | Same-codebase control is scientifically defensible | Locked |
| Keep thermal model labeled synthetic/untrained | No trained thermal artifact exists | Prevents false real-world thermal claim | Locked |
| Do not reuse incompatible HERMES checkpoints/metrics | Different state size, window, mobility, range, and evaluator | Avoids disconnected-simulator confound | Locked |
| Use full-slot idle model as primary plus header-only sensitivity | Both use inherited `E_elec`; effect size depends on awake duration | Phase 1 ordering 127/818/920 | Locked for sensitivity reporting |
| Rename “posterior” to state-conditioned transition probability | Transition row is not a Bayesian posterior | Terminology now matches implementation | Locked |
| Use rectified moments from HEART-CH equations | Harvest sampling is rectified | Removes mean/variance inconsistency | Locked |
| Use `T=24`, `n_max=3` | Derived from measured cluster contention | Creates meaningful scarcity | Locked primary environment |
| Use one packet/round, TTL=3, `q_max=5` | Makes sleep/service tradeoff measurable and bounded | Queue calibration and stale-drop metric | Locked |
| Use shared frozen schedule replay | Isolates MAC effects | CH/embedding co-adaptation deferred | Locked |
| Never repeat stale schedule frames | Repetition fabricates exogenous decisions | Right-censor at schedule exhaustion | Locked |
| Use dynamic identity-preserving curriculum | Fixed membership failed generalization | Training now matches evaluation structure | Locked |
| Enforce queue caps in action and target paths | Prevents infeasible actions and Q targets | Added tests and corrected agent | Locked |
| Stop idle-weight-10 run | Return support saturated without allocation correction | Broken-signal rule followed | Rejected run |
| Train budget-8 policy explicitly | Development sweep showed strong lifetime regime | Authoritative 500-episode gate passed | Locked as lifetime reference |
| Treat censored endpoints with KM/restricted time | No observed HTA deaths in schedule windows | Avoids invented death times | Locked |
| Remove old S2A2MAC novelty sentence | Source mechanism has per-node active layers | Novelty framing corrected | Locked |
| Delay Phase 4 | Primary QoS/lifetime operating point is unresolved | Avoids spending 210 runs on the wrong policy | Current recommendation |

---

## 11. Remaining doubts and risks

### 11.1 Primary operating point: budget 8 or budget 12

**Doubt:** Is the paper primarily about maximum lifetime/energy efficiency, or
about a balanced MAC policy that also preserves delivery fairness and packet
freshness?

**Current evidence:** Budget 8 is authoritative and survives the full observed
schedule windows, but its held-out stale-drop count is very high. A
development-only budget-12 inference sweep showed a much more balanced profile,
but it used a non-authoritative 200-episode checkpoint.

**Recommendation:** Keep budget 8 as the lifetime-optimized reference. Train a
new schedule-matched budget-12 candidate for 500 episodes and evaluate it only
on development seeds. Do not inspect the held-out pilot seeds again during
tuning.

### 11.2 Acceptance criteria for a balanced candidate

The following development criteria are recommended before training so the
decision is not made after seeing favorable outputs:

```text
median FND              >= 200 rounds
median throughput       >= 22,500 packets
median Jain fairness    >= 0.90
median stale drops      <= 47,000
always-sleep collapse   = false
reward domination       = false
Phase 2 convergence     = pass
```

These are proposed engineering criteria, not already validated scientific
thresholds. The instructor should approve or revise them before the balanced
candidate becomes authoritative.

### 11.3 Whether to add a stale-drop reward term

**Doubt:** The formal reward contains queue fairness but not an explicit stale
drop or delay penalty.

**Risk:** Adding a new reward term now changes the formal problem and creates
another hyperparameter/ablation obligation.

**Recommendation:** First test whether budget-12 training satisfies the
predeclared QoS criteria without modifying the reward. Add a stale-drop penalty
only if that controlled attempt fails and the paper is explicitly revised to
include it.

### 11.4 Exogenous embedding limitation

The ST-GCN embeddings and CH schedules are replayed from the upstream policy.
They do not update in response to MAC-created energy trajectories.

This is intentional for causal attribution, but it limits claims about fully
closed-loop network co-adaptation. The paper should state:

> CH selection is evaluated under a fixed exogenous schedule to enable clean
> causal attribution of MAC-layer effects; joint co-adaptation of CH selection
> and MAC scheduling is left to future work.

### 11.5 Synthetic thermal model

The thermal source tests dual-source mechanism behavior, not real thermal
forecasting. Solar-only versus hybrid ablation can establish whether the second
source changes policy behavior under the simulator, but cannot validate thermal
prediction against a physical trace.

### 11.6 Idle-duration interpretation

The full-slot model is dimensionally consistent with receiving for a full
4000-bit slot, but it produces a dramatic lifetime change. The header-only
variant must remain visible in the Phase 5 sensitivity results. The manuscript
must not imply that one duration universally represents every sensor radio and
TDMA synchronization design.

### 11.7 Schedule-censor boundary

Frozen HEART-CH schedules end around 1600-1700 rounds because the upstream
policy can no longer select a CH. This is a legitimate observation boundary,
but it prevents direct observation of HTA-MAC’s eventual FND/HND under the
budget-8 policy.

The paper must report the censoring rate and use survival methods. It must not
say “infinite lifetime” or silently substitute 3000 rounds.

### 11.8 Analytical lifetime bound

The proposed expression

```text
L(i,k) = E_i(t) / (E_consumed - mu_k)
```

has not yet been established as a proof. The denominator sign, transition
assumptions, time variation, battery cap, and stochastic harvest must be
handled formally. The Phase 5 requirement of at least 95% empirical coverage
has not been tested.

### 11.9 Novelty and citation risk

The earlier closest-competitor statement was already corrected after source
inspection. This shows that the complete novelty claim and every DOI must still
be checked against primary sources before submission. Existing README or
prompt wording is not evidence of novelty.

### 11.10 Repository state

The parent worktree contains unrelated modified and untracked files, and the
entire `hta-mac/` directory is currently untracked from the parent repository’s
perspective. No commit or publication snapshot has been created in this work.

Before final experiments, the HTA-MAC code and frozen configuration should be
placed in a dedicated tracked repository or intentionally committed on a
project branch. Unrelated parent changes must not be included.

---

## 12. Recommended next execution sequence

### Step 1: make policy selection configurable

Add command-line options to `experiments/run_phase3_pilot.py` for:

```text
--hta-checkpoint
--hta-budget
```

This prevents manual source edits when comparing budget-8 and budget-12
checkpoints and makes the exact evaluated policy visible in the command and
summary.

### Step 2: train one balanced budget-12 candidate

Use the same:

- dynamic schedule-matched curriculum;
- training seeds 2300-2304;
- five cluster ranks;
- environment capacity `T=24`;
- `n_max=3`;
- reward weights;
- learning frequency;
- queue caps;
- convergence and collapse gates.

Change only the internal projection cap from 8 to 12. Record whether training
starts from the tuned pre-authoritative checkpoint or from a fixed declared
initialization; do not choose between initializations after comparing held-out
results.

### Step 3: validate only on development seeds

Use seeds 2400-2404 and apply the predeclared balanced-candidate acceptance
criteria. Do not use seeds 3100-3104 for model selection.

### Step 4: freeze the primary checkpoint

- If budget 12 passes, use it as the balanced primary model and retain budget 8
  as the lifetime-optimized ablation/reference.
- If budget 12 fails, either retain budget 8 and frame the paper explicitly as
  a lifetime/energy tradeoff, or formally revise the reward to include packet
  freshness and repeat Phase 2. Do not hide the failure.

### Step 5: run Phase 4 once

After the checkpoint and criteria are frozen:

- use 30 new paired seeds, recommended 4000-4029;
- run all seven policies;
- archive all raw per-trial rows;
- report median and IQR;
- report event counts and censoring;
- use paired Wilcoxon tests on valid paired endpoints;
- include censor-aware lifetime analysis;
- record all crashes, NaNs, and exclusions.

### Step 6: Phase 5 ablations

Run:

- `n_max in {1,2,3,4}`;
- reward-weight sensitivity for `w4` and `w5`;
- solar-only versus hybrid source;
- full-slot idle versus header-only versus idle-off;
- frame-budget sensitivity around the measured contention regime;
- analytical bound validation against logged declining-state trajectories.

### Step 7: Phase 6 manuscript

Only after Phase 4 and Phase 5 archives are frozen:

- generate figures directly from archived CSVs;
- generate notation, setup, reward, main-result, and ablation tables;
- verify every DOI;
- complete the primary-source novelty audit;
- use IMRAD structure;
- state the thermal and exogenous-schedule limitations prominently;
- ensure every number maps to an archived row or summary.

---

## 13. Files implemented or materially added

### Core/frozen components

- `core/configuration.py`
- `core/reproducibility.py`
- `core/frozen_assets.yaml`
- `core/ch_selection/frozen_heart_ch.py`
- `core/ch_selection/frozen_schedule.py`
- `core/ch_selection/frozen_schedule_full.py`
- `core/energy/idle_model.py`
- `core/energy/radio_model.py`
- `core/hmm/solar_hmm.py`
- `core/hmm/thermal_hmm.py`
- `core/hmm/rectified_moments.py`

### Environments

- `envs/intra_cluster_mac_env.py`
- `envs/scheduled_mac_env.py`
- `envs/fixed_cluster_training_env.py`
- `envs/dynamic_cluster_training_env.py`
- `envs/idle_isolation_env.py`

### Agent and policies

- `agents/branching_dqn.py`
- `agents/prioritized_replay.py`
- `agents/reward_model.py`
- `agents/budget_projection.py`
- `baselines/interface.py`
- `baselines/policies.py`

### Experiments and analysis

- `experiments/train_phase2_fixed_cluster.py`
- `experiments/train_phase2_curriculum.py`
- `experiments/train_phase2_dynamic_curriculum.py`
- `experiments/evaluate_phase2_fixed_cluster.py`
- `experiments/run_phase3_pilot.py`
- `validation/analyze_phase3_pilot.py`
- Phase 0/1 gate, calibration, and feature-validation scripts under
  `validation/`

### Documentation and provenance

- `PHASE0_STATUS.md`
- `PHASE1_STATUS.md`
- `PHASE2_STATUS.md`
- `PHASE3_STATUS.md`
- `PHASE2_3_REMEDIATION_REPORT.md`
- `BASELINE_PROVENANCE.md`
- `PRE_PHASE2_DECISION_CLOSURE.md`
- `paper/SCOPE_LIMITATIONS_BOILERPLATE.md`

---

## 14. Authoritative evidence index

| Purpose | Artifact |
|---|---|
| Corrected Phase 0 result | `outputs/logs/phase0_corrected_gate.json` |
| Authoritative Phase 1 gate | `outputs/logs/phase1_gate.json` |
| Feature equivalence/semantics | `outputs/logs/harvest_feature_validation.json` |
| Cluster contention | `outputs/logs/cluster_contention_analysis.json` |
| Queue calibration | `outputs/logs/queue_capacity_calibration.json` |
| Reward calibration | `outputs/logs/phase2_reward_calibration.json` |
| Phase 2 training summary | `outputs/phase2/authoritative_dynamic_budget8_500ep/summary.json` |
| Phase 2 episode log | `outputs/phase2/authoritative_dynamic_budget8_500ep/episodes.jsonl` |
| Phase 2 checkpoint | `outputs/phase2/authoritative_dynamic_budget8_500ep/branching_c51.pt` |
| Development budget sweep | `outputs/phase2/development_budget_sweep_tuned_idle3.json` |
| Independent development validation | `outputs/phase3/development_validation_authoritative_budget8/summary.json` |
| Held-out raw pilot trials | `outputs/phase3/heldout_pilot_authoritative_budget8_censor_aware/raw_trials.csv` |
| Held-out censor-aware summary | `outputs/phase3/heldout_pilot_authoritative_budget8_censor_aware/summary.json` |
| Baseline mechanism provenance | `BASELINE_PROVENANCE.md` |
| Immutable pre-Phase-2 archive | `outputs/archive/authoritative_pre_phase2_20260728/` |

Held-out raw CSV SHA-256:

```text
7529E8CCB99F35F6E0BCC6552B62BD9C5B8DE95B2E584548E17C51FD6D076438
```

---

## 15. Verification command

From:

```text
F:\WSN\matlab\stage2\hta-mac
```

run:

```powershell
python -B -m pytest validation -q -p no:cacheprovider
```

Last verified result:

```text
25 passed
49 upstream library deprecation warnings
0 failures
```

---

## 16. Current honest conclusion

HTA-MAC now has a defensible implementation foundation, explicit energy and
queue accounting, frozen upstream control, a functioning Branching DQN,
schedule-matched training, a passed Phase 2 checkpoint, common-interface
baselines, and a complete five-seed held-out pilot.

The work has also exposed the paper’s central unresolved question: how much
packet freshness and fairness may be traded for lifetime and idle-energy
savings. The budget-8 checkpoint demonstrates the lifetime-focused extreme,
but its stale-drop behavior prevents an unconditional superiority claim.

The correct next move is not to launch 210 final runs immediately. It is to
predeclare a balanced operating criterion, train and development-test one
budget-12 candidate, freeze the primary policy, and then run Phase 4 once on
untouched seeds.

Until Phase 4, Phase 5, the analytical-bound validation, and the citation audit
are complete, this project should be described as a rigorously validated
research prototype with promising pilot evidence—not as a finished,
publication-ready paper.
