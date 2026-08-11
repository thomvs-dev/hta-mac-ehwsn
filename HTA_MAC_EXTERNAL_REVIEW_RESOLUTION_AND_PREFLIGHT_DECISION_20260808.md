# HTA-MAC external-review resolution and preflight decision

**Prepared:** 8 August 2026  
**Decision status:** resolved before fresh 5399/6399/7399 Colab training  
**Scope:** paper-aligned B16 QoS-repaired side study

## Executive answer

The checkpoint-producing architecture is **`EquivariantSetBranchingC51`**, selected by `architecture="equivariant_set_branching"`. It is not the failed flattened/node-specific-head `GlobalBranchingDuelingC51`. The flattened architecture remains retired; that decision was not reversed.

The paper-aligned B16 profile is a **secondary literature-alignment side study**, not a replacement for the primary registered idle-on, hybrid solar+thermal, frozen HEART-CH track. Consequently, this B16 profile cannot support the primary C1 hybrid-harvesting or C3 idle-listening claims.

A new 300-round preflight measurement shows that B=16 is not persistently binding for the successful policies. HTA-MAC used a median 31.23% of available budget and encountered feasible queue demand above B in only 1.33% of active cluster-rounds. Therefore, its 300-round throughput result is evidence of service feasibility/quality in this paper-aligned profile, not strong evidence under persistent slot scarcity.

Training may proceed only under those boundaries and with the newly added budget-utilization/contention reporting. Confirmation seeds remain unauthorized.

## 1. Architecture question: direct resolution

### 1.1 Current architecture

Current code maps:

```text
architecture="equivariant_set_branching"
    -> agents.architectures.EquivariantSetBranchingC51
```

The selected architecture has:

- one shared node encoder;
- masked mean and masked max aggregation;
- an invariant global context;
- one shared advantage head for every physical node;
- no learned raw node identity;
- parameter count independent of maximum branch capacity.

The rejected architecture maps:

```text
architecture="shared_branching"
    -> agents.architectures.GlobalBranchingDuelingC51
```

It still exists only for historical compatibility and registered ablations. It is not selected by the repaired Colab command.

### 1.2 Evidence from the previous B16 checkpoints

All three old B16 summaries—optimizer seeds 5299, 6299, and 7299—record:

```text
architecture = equivariant_set_branching
state_schema = phase2d_ttl_cap_v2
input_dim = 58
trajectory_loss_weight = 1.0
concavity_loss_weight = 0.1
```

Their stored foundation audits all report `gate_pass`, random projected-allocation agreement 1.0, maximum projected-allocation L1 error 0, and no feasibility failure.

### 1.3 Fresh audit using current checkpoint-producing code

The seed-5299 checkpoint was re-audited under the current code and repaired paper-aligned profile before authorizing another GPU run.

Evidence:

`outputs/audits/paper_aligned_b16_current_code_preflight_foundation_seed5299_20260808.json`

Results:

| Audit | Result |
|---|---:|
| Status | `gate_pass` |
| Random permutations | 20 |
| Targeted swaps | 10 |
| Maximum random log-probability error | 4.7684e-7 |
| Maximum random Q error | 8.9407e-7 |
| Random local-argmax agreement | 1.0 |
| Random projected-allocation agreement | 1.0 |
| Maximum projected-allocation L1 | 0 |
| All allocations budget/cap feasible | true |
| Q values within one atom of C51 boundary | 0.0 |

This explicitly re-passes the identity/equivariance gate. The previous 53–58% agreement result belongs to the retired flattened architecture and does not describe the current network.

## 2. Paper-aligned B16 track decision

The profile now explicitly records:

```text
track_role = secondary_literature_alignment_side_study
primary_track_replaced = false
primary_track = registered_idle_on_hybrid_solar_thermal_frozen_heart_ch
primary_contributions_evaluated_by_this_profile = false
```

This side study intentionally disables:

1. idle-listening energy accounting, associated with contribution C3; and
2. thermal harvesting, part of the hybrid solar+thermal contribution C1.

It also moves the base station to `(50,50)` and uses a 500-bit control packet to improve contextual alignment with Hasani et al. and Ge et al. These choices are valid for a secondary comparison environment but cannot silently replace the primary experiment.

Decision:

- keep the original idle-on, hybrid-harvest, BS=`(50,175)` track as the primary contribution track;
- run repaired B16 only as a literature-positioning, service-quality, energy, and lifetime side study;
- never use B16-side-study results alone to claim C1 or C3.

## 3. Does B=16 create real contention?

### 3.1 Added measurements

`experiments/run_phase3_pilot.py` now reports per policy:

- mean allocated slots per active cluster-round;
- mean feasible queued demand slots per active cluster-round;
- allocated-budget utilization fraction;
- fraction of active cluster-rounds whose allocation reaches B;
- fraction of active cluster-rounds whose feasible queued demand exceeds B.

Feasible demand is calculated before the action as:

```text
sum(min(queue_i, n_max)) over alive non-CH members
```

This distinguishes a policy choosing not to fill the frame from a frame that is genuinely too small for queued demand.

### 3.2 Measured preflight result

The old equivariant seed-5299 checkpoint and all bundled comparators were replayed for 300 rounds on development seeds 2400–2404 using the repaired profile. All 35 policy/seed runs completed with finite metrics.

Evidence directory:

`outputs/phase3/paper_aligned_b16_budget_pressure_audit_seed5299_20260808/`

Median across five schedules:

| Policy | Allocated slots / active cluster-round | Feasible demand | Budget utilization | Allocation reaches B | Demand exceeds B | Delivery |
|---|---:|---:|---:|---:|---:|---:|
| HTA-MAC | 4.997 | 5.099 | 31.23% | 1.73% | 1.33% | 0.9960 |
| Energy-proportional | 9.878 | 5.166 | 61.74% | 25.60% | 1.43% | 0.9960 |
| Harvest-proportional | 9.878 | 5.556 | 61.74% | 25.60% | 2.08% | 0.9938 |
| Random-budgeted | 9.878 | 5.541 | 61.74% | 25.60% | 2.10% | 0.9949 |
| Static equal | 3.995 | 11.867 | 24.97% | 0.33% | 25.12% | 0.7962 |
| FFSS-adapted | 3.995 | 11.867 | 24.97% | 0.33% | 25.12% | 0.7962 |
| S2A2MAC-adapted | 4.068 | 9.897 | 25.42% | 6.03% | 17.77% | 0.7684 |

### 3.3 Interpretation

B=16 is not a persistent scarcity constraint for HTA-MAC, energy-proportional, harvest-proportional, or random-budgeted policies. Their successful service keeps queued demand low; demand exceeds B in only about 1–2% of active cluster-rounds.

Static/FFSS and S2A2MAC-adapted accumulate much more backlog and experience contention more frequently, but primarily because their allocation rules under-serve traffic—not because all policies are continuously pressing against B.

Therefore:

- the 300-round result can show whether a policy keeps up with offered traffic;
- it cannot be presented as a strong persistent-contention comparison;
- near-equal HTA/energy-proportional throughput is consistent with a service ceiling;
- the action-distinctness and energy/lifetime evidence are more informative than another throughput percentage in this profile.

A future persistent-contention experiment should be a separate, preregistered stress profile, for example a lower B or validated higher burst load. It must not replace or retroactively alter this literature-aligned B16 side study.

## 4. Open mechanism items

### 4.1 Death penalty

The death reward weight is present (`w3=2.0` in the old B16 summaries), but it was structurally inert in the 300-round B16 training environment:

| Seed | Logged steps | Death events | Steps firing death term | Death/packet magnitude ratio |
|---:|---:|---:|---:|---:|
| 5299 | 150,000 | 0 | 0 | 0.0 |
| 6299 | 150,000 | 0 | 0 | 0.0 |
| 7299 | 150,000 | 0 | 0 | 0.0 |

It is incorrect to describe this as a meaningful gradient signal in B16 training. The 3,000-round Phase 3 evaluation may expose deaths, but it does not retroactively activate the 300-step training reward. If lifetime learning is required, a future training curriculum must include scientifically justified near-depletion states or a longer horizon as a new experiment version.
This zero-firing result is specific to the idle-listening-disabled B16 side profile. Idle listening was previously a major depletion mechanism in the primary track, so the result must not be generalized into a claim that the death reward is inherently inactive under the primary idle-on configuration.


### 4.2 Trajectory-order and concavity losses

Both mechanisms survived the architecture rebuild and are frozen in the repaired notebook:

```text
trajectory_loss_weight = 1.0
concavity_loss_weight = 0.1
trajectory_margin_fraction = 0.05
```

They are implemented in `BranchingDQNAgent.learn` and contribute to the total loss alongside C51. The earlier 79.63% marginal-ordering result is not automatically transferred to a new checkpoint; each fresh lineage must remeasure mechanism behavior.

### 4.3 Rank/percentile-within-active-cluster features

They are **not present** in `phase2d_ttl_cap_v2`.

Current 58-feature layout:

- 18 physical features;
- 4 packet-age histogram features;
- 4 action-validity/cap features;
- 32 frozen spatial-embedding features.

The equivariant network sees local node features plus masked mean/max context, but no explicit within-active-cluster rank or percentile. This is explicitly deferred to a separately versioned ablation after the repaired QoS run. Adding it now would combine a controller-accounting repair with an observation/architecture change and make attribution impossible.

If fresh results show action collapse, low target fairness, or inability to distinguish relative scarcity, the next experiment should add rank/percentile features under a new observation schema, with new permutation tests, scale calibration, optimizer seeds, and checkpoints.

## 5. Changes made in response to the review

1. Added a frozen architecture-decision artifact:
   `config/paper_aligned_hasani2025_architecture_decision_repaired.json`.
2. Marked the paper-aligned profile as a secondary side study and explicitly listed C1/C3 as disabled.
3. Re-ran the current-code foundation audit before GPU authorization.
4. Added budget-utilization, budget-binding, and demand-contention metrics to Phase 3.
5. Added regression tests for pressure accounting and architecture/track decisions.
6. Measured the existing seed-5299 checkpoint over all five development schedules and seven policies.
7. Carried the new metrics into the Colab source bundle.
8. Recorded the death penalty as inactive, the mechanism losses as retained, and rank features as deferred.

## 6. Authorization decision

The repaired Colab run is authorized as a **secondary side-study development run** because:

- the current architecture is equivariant;
- the current code re-passed Q-level and projected-allocation permutation gates;
- the old flattened architecture remains retired;
- B16's low-contention limitation is now measured and disclosed;
- the notebook will report allocation pressure for every fresh policy run;
- the primary track and C1/C3 boundaries are explicit.

The run is not authorized to support:

- primary-track replacement;
- C1 or C3 claims;
- persistent-contention superiority;
- lifetime superiority without observed/censor-aware evidence;
- confirmation or publication claims.

## 7. Recommended sequence after fresh results

1. Verify repaired cohort accounting and all architecture gates.
2. Recompute budget pressure for the fresh checkpoint.
3. Use the action audit to determine whether the energy-proportional tie is a decision tie or an environment ceiling.
4. Analyze 3,000-round censor-aware lifetime and energy efficiency.
5. Choose only one next experiment:
   - rank/percentile observation ablation if relative competition is not learned;
   - energy/marginal-slot regularization if excess allocation hurts packets/J;
   - a preregistered lower-B or burst-load stress profile if persistent contention evidence is needed;
   - longer/near-depletion training only if the lifetime objective must provide a gradient.
6. Keep confirmation seeds unused until one checkpoint and the full analysis are frozen.
