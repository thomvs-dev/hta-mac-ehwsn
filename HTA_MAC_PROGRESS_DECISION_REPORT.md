# HTA-MAC Progress, Decision, and Open-Issues Report

**Project:** HMM-Trajectory-Aware MAC (HTA-MAC)  
**Repository:** `F:\WSN\matlab\stage2\hta-mac`  
**Report date:** 2026-07-28  
**Current verified boundary:** Phase 0 and Phase 1 gates executed; Phase 2
training has not started.

---

## 1. Purpose of this report

This document is a technical handoff for the instructor and the next research
session. It records:

1. what was inspected, implemented, and executed;
2. the evidence produced by real runs;
3. the engineering rationale behind the main decisions;
4. failed or rejected approaches;
5. known implementation limitations and inconsistent files;
6. questions that must be resolved before Branching DQN training;
7. the remaining work from Phases 2 through 6.

This is not a manuscript draft and does not claim that HTA-MAC outperforms any
baseline. No learned MAC agent has been trained or evaluated.

“Reasoning” in this report means reproducible engineering rationale tied to
code, artifacts, or experimental observations. It does not treat undocumented
internal deliberation as scientific evidence.

---

## 2. Original scope retained

The work was kept within the bounded HTA-MAC problem:

- HEART-CH remains the upstream cluster-head mechanism.
- The HEART-CH checkpoint is frozen and never retrained.
- HTA-MAC controls only intra-cluster TDMA slot allocation.
- Routing and Pointer Networks were not implemented or modified.
- HERMES results were not merged with HEART-CH or HTA-MAC results.
- No performance improvement was fabricated.

The inherited system constants are stored in `config/base.yaml`. The intended
model remains 100 nodes in a 100 m × 100 m field, BS at `(50,175)` m,
`E0=0.5 J`, 4000-bit packets, five CHs at full network, `W=10`, `F=31`,
50 m transmission range, eight solar states, and four thermal states.

---

## 3. Repository and artifact inspection

### 3.1 Authoritative upstream repository

The clean upstream source used for frozen HEART-CH assets is:

```text
F:\WSN\matlab\stage2\final_repo
```

Verified upstream commit:

```text
d96abce25237feb2b6d6c660f6b4d605feb94330
```

Frozen checkpoint:

```text
outputs/checkpoints/model_v91_throughput.pt
SHA-256:
ccb572901e263a50954c9a9b0746cf596193d0018502e0c7b26b623e6d287c5f
```

Solar Stage 1 artifact:

```text
outputs/stage1_params.mat
SHA-256:
8a864e9e10235037fb86b71fc6cc3a35c9a8637593c965c797318eaeff52252c
```

The solar artifact contains the trained eight-state HMM parameters.

### 3.2 Thermal-model finding

No trained four-state thermal HMM export or thermal training trace was found.
The upstream simulator constructs thermal parameters from fixed Python
defaults.

Those defaults were frozen into:

```text
core/hmm/thermal_auxiliary_params.npz
SHA-256:
3d22e56e47a499884b50f88d2598124153ab99386ad62c327835d50cbc46c845
```

Its provenance is explicitly:

```text
synthetic_auxiliary_from_heart_ch_defaults
trained: false
```

Decision rationale: freezing the defaults makes runs reproducible, but it does
not convert them into trained parameters. Any manuscript wording that implies
both harvest HMMs were learned from data would currently be false.

### 3.3 HERMES inspection

The `hermes/` directory was inspected for reusable artifacts. It contained
useful diagnostics and reporting patterns but no compatible replacement for
the inherited HEART-CH checkpoint or a trained thermal HMM.

Rejected substitutions included:

- an analytical `heart_fixed` comparator rather than the neural checkpoint;
- a learned checkpoint configured for `W=20`, `F=33`, a different environment,
  and quorum-cover selection;
- a standalone simulator using 200 m maximum range and reflective mobility;
- thermal transitions generated from an assumed persistence parameter;
- learned evaluation explicitly marked `hermes_included=false`.

Decision rationale: adopting these artifacts would recreate the earlier
disconnected-simulator problem and invalidate paired comparisons.

Evidence:

```text
HERMES_ARTIFACT_AUDIT.md
outputs/logs/hermes_artifact_audit.json
```

---

## 4. Phase 0: foundation and reproducibility

### 4.1 Initial gate result

The original requirement was to reproduce:

```text
T_FND = 1191.3 ± 40.0 rounds
```

A fresh 30-trial run using seeds 1000–1029 produced:

```text
mean T_FND             = 1100.6 rounds
population std         = 44.18189674516023 rounds
median T_FND           = 1100.0 rounds
IQR                    = 58.25 rounds
NaN/crash trials       = 0
```

The original reproduction gate therefore failed.

Additional checks did not recover 1191.3:

- manuscript seeds 42–61 produced `1083.4 ± 37.13`;
- checkpoint metadata reported a ten-episode mean T_FND of `1122.1`;
- the locked summary lacked raw trials, a source commit, and a configuration
  hash sufficient to reproduce 1191.3.

### 4.2 Corrected-foundation decision

After the discrepancy and HERMES audit were reported, the user directed the
work to continue using the actual available artifacts. A corrected Phase 0
acceptance contract was created:

```text
config/phase0_acceptance.yaml
```

The corrected empirical foundation is:

```text
Frozen HEART-CH baseline: 1100.6 ± 44.18 over 30 trials
Thermal model: fixed synthetic auxiliary, trained=false
Retired reference: 1191.3 ± 40.0
```

The corrected gate then passed:

```text
PHASE_0_CORRECTED_GATE=PASS
```

Evidence:

```text
outputs/logs/phase0_corrected_gate.json
outputs/logs/phase0_gate_20260727T210959Z.json
validation/phase0_gate_corrected.py
```

### 4.3 Phase 0 code added

- restricted configuration loader;
- reproducibility utilities and artifact hashing;
- immutable solar-HMM loader;
- immutable thermal-auxiliary loader;
- exact first-order radio model;
- evaluation-only frozen HEART-CH adapter;
- HERMES artifact auditor;
- Phase 0 validation scripts and regression tests.

### 4.4 Important Phase 0 inconsistency still present

`PHASE0_STATUS.md` still describes the original failed gate and says not to
begin Phase 1. That document is historically accurate but operationally stale.
It must either be renamed as the original-gate report or updated with a clear
supersession notice.

`config/base.yaml` also still stores 1191.3 as the evaluation reference. The
corrected acceptance file retires it, but leaving both values without a single
authoritative resolution is risky.

Required cleanup before manuscript work:

1. preserve the original failure as provenance;
2. mark it explicitly superseded by `phase0_acceptance.yaml`;
3. prevent scripts from accidentally reading the retired reference.

---

## 5. Phase 1: structural frame

### 5.1 Idle-listening model

Implemented:

```text
core/energy/idle_model.py
```

The energy update is:

```text
E(t+1) = max(0, min(Emax, E(t) - C(t) + H(t)))
```

Idle energy is implemented using the mandated inherited electronics constant:

```text
E_idle = idle_slots × E_elec × slot_bit_times
```

For the Phase 1 gate:

```text
E_elec        = 5.0e-8 J/bit
slot duration = 4000 bit-times
one idle slot = 2.0e-4 J
```

Decision rationale: `E_elec` is an energy-per-bit constant, not a power in
watts. Expressing TDMA slot duration in packet bit-times makes the multiplication
dimensionally interpretable without inventing a new radio constant.

Major caution: this convention produces a very large idle cost. It satisfies
the supplied rule but needs explicit instructor approval and sensitivity
analysis before publication interpretation.

### 5.2 Intra-cluster environment

Implemented:

```text
envs/intra_cluster_mac_env.py
```

Current structural behavior includes:

- per-node energy and alive state;
- solar and thermal state evolution;
- rectified hybrid harvesting;
- member-to-CH transmission cost;
- CH reception, aggregation, and BS transmission cost;
- explicit idle-listening cost;
- per-node queues;
- discrete slot actions from zero to `n_max`;
- per-cluster budget enforcement;
- 18-dimensional per-node state;
- deterministic seeded reset and transition behavior.

The returned state shape is:

```text
[100, 18]
```

### 5.3 Budget projection

Implemented:

```text
agents/budget_projection.py
```

The projection exposes only the next feasible marginal Q gain for each node.
It therefore cannot assign action level two before assigning that node level
one. It stops when the budget is exhausted or all remaining marginal gains are
non-positive.

Validation result:

```text
1000 random cases
0 budget violations
```

### 5.4 Frozen HEART-CH integration

The first implementation used only one frozen CH decision at reset. This caused
a real diagnostic failure: the same fixed CHs died first from receive and
aggregation load in both idle-on and idle-off arms, masking the idle term.

That approach was rejected for the definitive gate.

The replacement generates a per-round CH schedule using the immutable upstream
checkpoint:

```text
core/ch_selection/frozen_schedule.py
envs/scheduled_mac_env.py
```

The same schedule is replayed in both paired arms. Consequently:

- CH selection is produced by the real frozen checkpoint;
- no CH parameters are retrained;
- both MAC policies receive identical exogenous clustering decisions;
- routing is unchanged;
- differences in the paired idle test cannot be caused by different CH
  schedules.

Decision rationale: a shared schedule provides experimental control and avoids
letting different MAC energy histories induce different upstream CH decisions.

Open interpretation issue: replaying an exogenous CH schedule also removes
feedback from HTA-MAC energy changes into later CH selections. The instructor
must decide whether “frozen upstream decision” means:

1. **shared schedule replay**, as currently implemented; or
2. **frozen policy evaluated online** using each policy's changed energy state.

Option 1 is cleaner for paired MAC attribution. Option 2 is more coupled but
allows CH-selection differences to confound the MAC comparison.

### 5.5 Phase 1 gate results

Definitive command:

```powershell
$env:PYTHONPATH='F:\WSN\matlab\stage2\hta-mac'
python -B validation\run_phase1_gate_scheduled.py
```

Verified printed results:

```text
FROZEN_CH_COUNT=5
ENERGY_TRACE_MAX_ERROR_J=0.000e+00
DETERMINISTIC=True
BUDGET_VIOLATIONS=0/1000
HMM_KS_SOLAR_D=0.0116, P=0.511606
HMM_KS_THERMAL_D=0.0157, P=0.169934
T_FND_IDLE_ON_OFF_MEDIAN=72.0/819.0
MEDIAN_SHIFT=747.0 rounds
PHASE_1_GATE=PASS
```

The five paired idle-ablation seeds were 2100–2104.

Raw network T_FND pairs:

| Seed | Idle on | Idle off | Difference |
|---:|---:|---:|---:|
| 2100 | 68 | 904 | 836 |
| 2101 | 78 | 877 | 799 |
| 2102 | 76 | 819 | 743 |
| 2103 | 72 | 798 | 726 |
| 2104 | 55 | 666 | 611 |

These are mechanism-validation results from five trials. They are not the
30-trial main comparison and must not appear as final performance evidence.

### 5.6 Interpretation of the KS tests

The KS tests compare independently sampled emissions from the frozen parameter
distributions.

They demonstrate that the new sampler is statistically consistent with the
inherited parameterized distribution. They do **not** prove agreement with a
real solar or thermal dataset.

This distinction is particularly important for the synthetic thermal model.

### 5.7 Phase 1 tests

Current focused regression result:

```text
6 passed
```

The tests cover:

- Phase 0 configuration and frozen HMM loading;
- idle-energy arithmetic;
- bounded energy update;
- marginal-level budget projection;
- leaving negative-gain budget unused.

This is not yet broad environment or end-to-end test coverage.

---

## 6. Failed attempts and what they revealed

### 6.1 Treating 1191.3 as reproducible

Result: failed on fresh and manuscript seed sets.

Lesson: the published/locked summary cannot remain the executable baseline
without its missing raw provenance.

### 6.2 Using HERMES artifacts as replacements

Result: rejected due incompatible window, feature count, CH-selection mode,
radio range, mobility, and disconnected evaluation.

Lesson: useful code or results are not valid substitutes merely because they
exist in the workspace.

### 6.3 Provisional 40-slot frame

Result: a real frozen cluster exceeded the 40-slot ceiling, making static equal
TDMA infeasible.

Action: the structural gate ceiling was increased to 100, which guarantees
feasibility for `N=100`.

Lesson: `T=100` is only a gate ceiling. It is not an optimized or
publication-ready slot budget.

### 6.4 One-shot fixed CH structure

Result: fixed CHs died first in both ablation arms, producing zero network FND
shift even while idle energy was accumulating.

Action: replaced by per-round frozen HEART-CH schedule replay.

Lesson: a working idle term can be hidden by an unrelated first-death
bottleneck. Inspection gates need to test causal observability, not only
whether a variable becomes nonzero.

### 6.5 Member-only isolation probe

A temporary diagnostic held CH receive and aggregation costs exogenous. It
produced an idle shift, proving the member idle calculation was active.

This probe was superseded by the full-accounting scheduled gate. It must not be
reported as the definitive Phase 1 result.

Definitive evidence is only:

```text
validation/run_phase1_gate_scheduled.py
outputs/logs/phase1_gate.json
```

---

## 7. Technical uncertainties requiring resolution before Phase 2

### 7.1 Meaning and units of `P_idle=E_elec`

Current implementation interprets one slot as 4000 bit-times and charges
`E_elec` per bit-time. This causes static TDMA T_FND to fall from a median 819
to 72 rounds in the five-trial gate.

Questions:

- Is a node intended to remain awake for every other member's full data slot?
- Does `T_idle` mean seconds, bit-times, or slot count?
- Is the very large effect acceptable as the intended accounting correction?
- Should control/guard/synchronization durations be modeled separately?

No new hardware constant should be introduced without evidence, but the current
interpretation must be explicitly accepted before it becomes the reward's idle
term.

### 7.2 “Posterior” versus transition-probability row

The 18-dimensional state currently uses the transition row associated with the
current discrete HMM state. This is normalized and trajectory-aware, but it is
not a Bayesian filtering posterior computed from a new observation sequence.

Questions:

- Should the paper call these values “state-conditioned next-state
  probabilities” rather than “posteriors”?
- Is there an upstream filtering posterior available that has not yet been
  located?
- If true posteriors are required, where are the per-round observations and
  forward-filter recursion?

The terminology must be corrected or the computation must be upgraded before
claim C1 is used.

### 7.3 Reuse of HEART-CH harvest features

The current environment reconstructs forecast mean, variance, and transition
rows from frozen HMM parameters and current states. It does not directly slice
the upstream 31-feature tensor.

This may conflict with the instruction to reuse the existing feature block
without recomputation.

Decision needed:

- extract the exact indices from HEART-CH's feature tensor; or
- formally validate numerical identity and document the adapter.

### 7.4 Rectified forecast moments

Actual harvest samples use `max(0, alpha*y)`. The forecast mean/variance in the
18-dimensional state currently use scaled Gaussian moments before
rectification.

For negative or near-zero state means, the expected rectified-normal harvest is
not equal to `alpha*mu`.

Decision needed:

- reproduce HEART-CH's existing approximation exactly; or
- use the correct rectified-normal moments and document that this is a new MAC
  feature definition.

The two choices should not be mixed silently.

### 7.5 ST-GCN embedding freshness

The frozen schedule archives the upstream ST-GCN embedding. Replaying it makes
the embedding independent of the changed HTA-MAC energy trajectory.

However, recomputing the embedding using changed MAC energy would require
constructing the exact upstream 31-feature, 10-round history while keeping the
encoder frozen.

Decision needed:

- replay embeddings as fixed upstream context; or
- recompute embeddings online with frozen weights and MAC-updated features.

The latter is more responsive but requires a carefully validated feature
adapter.

### 7.6 Schedule length

Frozen schedules are generated for at most 2000 upstream rounds. The MAC
environment permits 3000 rounds and currently reuses the last available
schedule frame after exhaustion.

This is acceptable only for the current Phase 1 FND probe because all reported
network FND events occurred before round 1000. It is not acceptable for final
T_HND, lifetime, or censored evaluations.

Required fix: generate schedules for the complete evaluation horizon or define
an explicit terminal/censoring rule.

### 7.7 Provisional `T` and `n_max`

Current structural settings:

```text
T gate ceiling = 100
n_max           = 3
```

`T=100` only prevents static infeasibility. It provides very weak resource
pressure and is not a defensible final TDMA frame design.

`n_max=3` follows the preliminary action design, but the planned ablation is
`{1,2,3,4}`.

Before training, a primary `T` must be selected and frozen. Otherwise agent
results may be dominated by an arbitrary budget.

### 7.8 Queue semantics

Each node currently generates one packet per round, sleeps can accumulate
packets, and multiple slots can drain a backlog.

Questions:

- Is one packet generated per alive node per round consistent with HEART-CH?
- Does each allocated slot always carry exactly one 4000-bit packet?
- Should stale packets expire?
- Does sleeping defer packets or count them as dropped?
- What is the justified value of `q_max=10`?

These choices directly affect throughput, fairness, and whether actions above
one slot are useful.

### 7.9 Reward weights

No reward weights have been selected or balanced. No Branching DQN has been
trained.

Weights `w1` through `w6` must be initialized using logged natural scales, not
chosen to force a favorable result. Always-sleep detection must be active from
the first training run.

### 7.10 Correct baseline for publication

The current verified frozen HEART-CH baseline is 1100.6 ± 44.18, not 1191.3 ±
40.0.

The instructor must decide whether to:

1. revise the inherited HEART-CH reference throughout the HTA-MAC study;
2. recover the exact missing configuration/checkpoint/raw trials for 1191.3;
3. report both values with a transparent reproducibility explanation.

The current implementation follows option 1 under user authorization, but the
manuscript has not yet been revised.

### 7.11 Thermal-harvest claim strength

The thermal auxiliary is synthetic and untrained. Consequently, current
experiments can test hybrid-source plumbing and controlled source ablations,
but they cannot establish real-world thermal forecasting validity.

The abstract and contribution language must not imply real thermal trace
training or validation.

---

## 8. Files that currently disagree or may mislead

| File | Current issue | Required action |
|---|---|---|
| `PHASE0_STATUS.md` | Says Phase 0 failed and Phase 1 must not begin | Mark as original gate and link corrected acceptance |
| `config/base.yaml` | Still contains retired 1191.3 reference | Remove from executable use or label historical |
| `config/mac_env.yaml` | Still a locked placeholder | Replace with authoritative Phase 1/2 config |
| `config/phase1.yaml` | Contains provisional 40-slot value | Retire or merge with resolved configuration |
| `config/phase1_gate.yaml` | Label says one-shot reset clustering | Update to per-round frozen schedule |
| `validation/run_phase1_gate.py` | Preliminary one-shot gate runner | Mark superseded |
| `validation/run_phase1_gate_isolated.py` | Temporary member-isolation runner | Mark diagnostic/superseded |
| `outputs/logs/phase1_gate.json` | Definitive current result | Preserve and hash before later runs overwrite it |

This cleanup should occur before Phase 2 so a future run cannot accidentally
use the wrong configuration or overwrite the only definitive gate evidence.

---

## 9. Work remaining

### Phase 2 — Branching DQN

Not started.

Required:

- freeze the decisions in Section 7;
- build shared trunk plus per-node dueling branches;
- decide how the frozen ST-GCN embedding is produced;
- reuse compatible replay/target/C51 infrastructure;
- implement reward component logging;
- train on a single fixed cluster;
- test for always-sleep collapse from the first run;
- show differentiated Q-values for equal-energy high- versus low-harvest
  states;
- pass the Phase 2 convergence gate with real logs.

### Phase 3 — baselines

Not started.

Required policies:

- static equal TDMA;
- energy proportional;
- harvest proportional;
- S2A2MAC adapted alternating-cluster sleep plus per-node 1/2/3 active layers;
- FFSS-style adapted scheduler;
- HTA-MAC;
- the seventh policy must be identified consistently because the blueprint says
  both “six baselines” and “7 policies.”

All must use one `MACPolicyInterface` and the same environment.

### Phase 4 — main evaluation

Not started.

Required:

- 30 paired independent trials per policy;
- raw per-trial CSV;
- median and IQR;
- paired Wilcoxon signed-rank test;
- T_FND, T_HND, throughput, idle waste, and queue fairness;
- explicit NaN/crash accounting.

### Phase 5 — ablations and analytical bound

Not started.

Required:

- `n_max ∈ {1,2,3,4}`;
- `w4/w5` reward sweep;
- solar-only versus hybrid;
- idle on versus off;
- empirical lifetime-bound validation;
- at least 95% coverage for sampled declining-state node-rounds.

The proposed bound still requires formal sign and denominator checks.

### Phase 6 — paper and handover

Not started.

Required:

- figures and tables generated only from archived data;
- DOI verification for every cited work;
- novelty language rechecked against primary sources;
- manuscript IMRAD sections;
- contribution claims limited to actual results;
- no claim that the synthetic thermal auxiliary was trained;
- no numeric comparison against materially different environments.

---

## 10. Recommended next actions

Before any neural-agent training:

1. resolve the idle-energy time/unit interpretation;
2. decide shared CH-schedule replay versus online frozen-policy evaluation;
3. decide replayed versus online-recomputed ST-GCN embeddings;
4. verify or replace the “posterior” terminology;
5. validate forecast moments against the exact HEART-CH feature computation;
6. select and freeze the primary frame budget `T`;
7. freeze queue-generation and packet-service semantics;
8. reconcile the stale configuration/status files;
9. archive the current Phase 0 and Phase 1 reports under unique timestamped
   names so they cannot be overwritten;
10. only then implement and train the Branching DQN.

Recommended default research choices, subject to instructor approval:

- retain a shared frozen per-round CH schedule for clean paired MAC attribution;
- recompute the frozen ST-GCN encoder online only after exact feature equivalence
  is proven;
- call transition rows “state-conditioned transition probabilities,” not
  posteriors, unless a true filter is added;
- keep the thermal model explicitly synthetic;
- treat 1100.6 ± 44.18 as the executable baseline unless the missing 1191.3
  provenance is recovered;
- run an idle-slot-duration sensitivity analysis before interpreting the large
  FND shift.

---

## 11. Exact current verification commands

From:

```text
F:\WSN\matlab\stage2\hta-mac
```

Phase 0:

```powershell
$env:PYTHONPATH='F:\WSN\matlab\stage2\hta-mac'
python -B validation\phase0_gate_corrected.py
```

Phase 1:

```powershell
$env:PYTHONPATH='F:\WSN\matlab\stage2\hta-mac'
python -B validation\run_phase1_gate_scheduled.py
```

Focused regression tests:

```powershell
$env:PYTHONPATH='F:\WSN\matlab\stage2\hta-mac'
python -B -m pytest validation -q -p no:cacheprovider
```

Last verified outputs:

```text
PHASE_0_CORRECTED_GATE=PASS
PHASE_1_GATE=PASS
6 passed
```

---

## 12. Evidence index

| Purpose | Artifact |
|---|---|
| Corrected Phase 0 acceptance | `config/phase0_acceptance.yaml` |
| Corrected Phase 0 result | `outputs/logs/phase0_corrected_gate.json` |
| Original 30-trial evaluation | `outputs/logs/phase0_gate_20260727T210959Z.json` |
| Frozen artifact manifest | `core/frozen_assets.yaml` |
| HERMES compatibility audit | `HERMES_ARTIFACT_AUDIT.md` |
| HERMES machine report | `outputs/logs/hermes_artifact_audit.json` |
| Idle energy implementation | `core/energy/idle_model.py` |
| MAC environment | `envs/intra_cluster_mac_env.py` |
| Frozen CH schedule | `core/ch_selection/frozen_schedule.py` |
| Scheduled environment | `envs/scheduled_mac_env.py` |
| Budget projection | `agents/budget_projection.py` |
| Definitive Phase 1 runner | `validation/run_phase1_gate_scheduled.py` |
| Definitive Phase 1 result | `outputs/logs/phase1_gate.json` |
| Phase 1 summary | `PHASE1_STATUS.md` |
| Focused tests | `validation/test_phase0_foundation.py`, `validation/test_phase1_primitives.py` |

---

## 13. Current honest conclusion

The project now has a reproducible corrected foundation, frozen artifact
provenance, a functioning intra-cluster MAC structural environment, explicit
idle accounting, constrained slot projection, frozen per-round CH scheduling,
and passing Phase 0/1 inspection evidence.

It does **not** yet have:

- a trained HTA-MAC agent;
- a validated reward;
- final frame-budget selection;
- publication-ready baselines;
- 30-trial comparative results;
- ablations;
- an empirically validated analytical bound;
- verified novelty or DOI coverage;
- a journal-ready manuscript.

The largest immediate risks are the interpretation of idle-energy units, the
use of transition rows as “posteriors,” the freshness of the reused ST-GCN
embedding, the synthetic thermal model, and the unresolved 1191.3 versus
1100.6 baseline discrepancy. Training before resolving these items would risk
producing a technically converged agent on an ambiguous problem definition.
