# HTA-MAC Phase 2D Foundation Implementation

**Prepared:** 4 August 2026  
**Repository:** `F:\WSN\matlab\stage2\hta-mac`  
**Implementation base HEAD:** `02c079ce6ee1eca19b0d23b750c92a01164a85eb`  
**Status:** Foundation gate passed; 50-episode learning smoke completed; full convergence and performance confirmation not yet authorized

---

## 1. Implemented outcome

Phase 2D now has an executable, tested replacement for the rejected branch-indexed model:

- versioned 58-feature policy observations with explicit TTL-age and action-validity information;
- `EquivariantSetBranchingC51`, using a shared node encoder, masked mean/max set context, a global value head, and one shared node-action advantage head;
- bundle-stable deterministic tie priorities in the hard-budget projector;
- checkpoint state-schema and embedding-boundary compatibility guards;
- trained-checkpoint permutation and C51-boundary audit tooling;
- focused property tests plus the unchanged legacy validation suite.

The frozen HEART-CH schedule, simulator accounting, reward weights, reward scale, C51 support, budget, queue capacity, and development seeds were not changed.

---

## 2. State schema

The base environment still emits the original 18 physical features. This preserves Phase 1 accounting and legacy Phase 2C reproduction. The Phase 2D training wrappers build a versioned policy observation:

| Block | Features | Positions |
|---|---:|---:|
| Physical/HMM state | 18 | 0-17 |
| Normalized packet-age histogram, ages 0-3 | 4 | 18-21 |
| Queue-feasible action validity, actions 0-3 | 4 | 22-25 |
| Frozen STGCN embedding | 32 | 26-57 |
| Total | 58 | 0-57 |

Schema identifier: `phase2d_ttl_cap_v2`.

The age bins distinguish equal-length queues with different expiry urgency. Counts are normalized by `q_max=5`. Invalid internal ages outside the retained TTL range raise an error. Input normalization now starts at the recorded embedding boundary, so age/cap features are not accidentally layer-normalized together with the inherited embedding.

Legacy wrappers default to `phase2c_v1`; the dynamic trainer selects Phase 2D automatically only for `--architecture equivariant_set_branching`.

---

## 3. Model architecture

Implemented class: `agents.architectures.EquivariantSetBranchingC51`.

```text
h_i    = shared_node_encoder(x_i)
mean   = masked_mean_i(h_i)
max    = masked_max_i(h_i)
g      = global_context(mean, max, active_fraction, normalized_budget)
V      = value(g)
A_i(a) = shared_advantage(h_i, g)
Z_i(a) = V + A_i(a) - mean_a A_i(a)
```

Properties:

- no flattening of node order;
- no `ModuleList` of node-specific advantage heads;
- no learned node-ID input;
- learned parameter count independent of maximum branch capacity;
- inactive branches receive a deterministic uniform categorical distribution;
- float64 accumulation is used only for the small invariant mean reduction so float32 permutation error remains below the frozen `1e-6` gate.

Measured parameter count: **115,123**, compared with **2,842,811** for the recorded Phase 2B shared-branching configuration. This is a measured architectural reduction, not yet evidence of improved final performance.

---

## 4. Hard-budget projection and symmetry breaking

`project_slot_budget` now accepts a unique `tie_break_priorities` vector. Heap ordering is:

```text
(-marginal_gain, physical_priority, branch_position, slot_level)
```

The physical priority is moved with the complete state/mask/cap/action-mask bundle during a permutation. It is not supplied as a learned feature. This makes exact discrete ties deterministic without teaching the network node identity.

The identity audit now transports these priorities and uses the Phase 2D observation schema recorded in each checkpoint.

---

## 5. Files added or changed

### Added

- `envs/policy_observation.py`
- `validation/test_phase2d_equivariant_foundation.py`
- `experiments/audit_phase2d_foundation.py`
- `config/phase2d_foundation.yaml`
- this implementation report

### Changed

- `agents/architectures.py`
- `agents/branching_dqn.py`
- `agents/budget_projection.py`
- `agents/branch_permutation.py`
- `envs/fixed_cluster_training_env.py`
- `envs/dynamic_cluster_training_env.py`
- `experiments/train_phase2_dynamic_curriculum.py`
- `experiments/audit_phase2c_branch_identity.py`

No file under the unrelated pre-existing `outputs/phase4/` directory was changed.

---

## 6. Correctness gates

Focused tests cover:

- TTL-age observability and invalid-age rejection;
- action-validity encoding;
- 20 deterministic node permutations;
- log-probability and Q equivariance at `<=1e-6`;
- cross-node global context;
- parameter-count independence from branch capacity;
- bundle-stable exact-tie allocation;
- embedding-only normalization;
- one finite prioritized-replay learning update.

The real frozen-schedule integration smoke produced:

```text
environment count: 5
observation shape: (100, 58)
active branches: 24
allocation budget/cap feasible: true
all Q-values finite: true
online parameters: 115123
```

---

## 7. CLI wiring smoke

Artifact:

`outputs/phase2/phase2d_equivariant_wiring_smoke_20260804`

This one-episode, five-step run verified end-to-end CLI selection, observation construction, action projection, checkpoint serialization, and config persistence. Its Phase 2 curriculum gate is deliberately false because it did not cross replay warmup.

---

## 8. Seed-2299 learning-mechanics smoke

Artifact:

`outputs/phase2/phase2d_equivariant_learning_smoke_seed2299_50ep_20260804`

Command:

```powershell
python -B experiments\train_phase2_dynamic_curriculum.py `
  --episodes 50 `
  --max-steps 300 `
  --development-seeds 2300,2301,2302,2303,2304 `
  --optimizer-seed 2299 `
  --run-name phase2d_equivariant_learning_smoke_seed2299_50ep_20260804 `
  --architecture equivariant_set_branching `
  --projection-budget 12 `
  --reward-scale-config config\phase2c_return_scale.json `
  --learning-rate 1e-5 `
  --normalize-input-blocks `
  --trajectory-loss-weight 1.0 `
  --concavity-loss-weight 0.1 `
  --precision fp32
```

Measured facts:

| Metric | Result |
|---|---:|
| Episodes | 50 |
| Environment steps | 7,875 |
| Curriculum pairs | 25 |
| Complete pair coverage | true |
| Pair cycles | 2 |
| Replay warmup crossed | true |
| Nonfinite stop | false |
| Always-sleep collapse | false |
| Pathological reward domination | false |
| Greedy mean packets | 1,548.8 |
| S8-vs-S1 maximum absolute Q difference | 0.00741929 |

The full Phase 2 curriculum gate is **false**, not because a foundation gate failed, but because 50 episodes provide only one stability snapshot and cannot satisfy the existing convergence/stability protocol. This run must not be described as a converged or publication candidate policy.

Checkpoint SHA-256:

`DCBBE0F65253ADEBA0D520FD0F66548AC393669375286ABEEEE52DE218960AB3`

---

## 9. Trained-checkpoint foundation audit

Artifact:

`outputs/phase2/phase2d_equivariant_learning_smoke_seed2299_50ep_20260804/phase2d_foundation_audit.json`

Audit scope:

- 20 deterministic active-branch permutations;
- 10 targeted active-branch swaps;
- complete state/mask/cap/action-mask/tie-priority movement;
- inverse-mapped logits, Q-values, local actions, and projected allocations;
- development-only C51 boundary occupancy.

Measured results:

| Gate metric | Result | Threshold |
|---|---:|---:|
| Maximum random log-probability error | `4.7683716e-7` | `<=1e-6` |
| Maximum random Q error | `8.9406967e-7` | `<=1e-6` |
| Random projected allocation agreement | `1.0` | `>=0.95` |
| Targeted projected allocation agreement | `1.0` | `>=0.90` |
| Local argmax agreement | `1.0` | `=1.0` |
| Budget/cap feasibility | true | required |
| Median bottom+top atom mass | `0.04057877` | `<0.10` |
| Q within one atom of either boundary | `0.0` | `<=0.05` |
| Q 1% / 50% / 99% | `-0.3321 / 0.2370 / 0.3712` | diagnostic |

Overall trained-checkpoint foundation audit: **gate_pass**.

---

## 10. Validation

Primary command:

```powershell
python -B -m pytest validation -q -p no:cacheprovider
```

Final verification: **73 passed, 93 warnings**. All warnings are the existing third-party PyTorch/PyG deprecation warnings; no HTA-MAC test failed.

---

## 11. What remains deliberately unimplemented

The staged plan forbids adding all changes at once. Therefore this foundation does **not** yet implement:

- a stale-drop/delivery/fairness Lagrangian or virtual-queue constraint;
- PopArt or IQN;
- attention or a GNN;
- network-wide final development selection;
- three-lineage confirmation;
- untouched 30-seed evaluation.

These are not omissions from the foundation gate. Objective changes come next only after the representation and learning mechanics are valid, which they now are.

---

## 12. Next authorized decision

Do not begin 125/500-episode three-lineage confirmation yet. The next step is Phase 2D-4 development attribution:

1. freeze the current foundation checkpoint and audit hashes;
2. define development-only delivery, stale-drop, and fairness constraints without accessing held-out seeds;
3. implement the constraint update and its tests;
4. compare the frozen reward with the constrained objective on development schedules;
5. require the identity and C51 boundary gates to continue passing;
6. only then authorize longer convergence runs and the other optimizer lineages.

This preserves causal attribution: the measured foundation result belongs to state sufficiency plus equivariant representation, not to an unrecorded simultaneous reward redesign.
