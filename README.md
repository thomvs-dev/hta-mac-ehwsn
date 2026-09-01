# HTA-MAC

HTA-MAC is a research framework for constrained intra-cluster medium-access
control in energy-harvesting wireless sensor networks. It uses a shared,
permutation-equivariant branching distributional controller to allocate a
bounded slot budget while accounting for delivery, packet staleness, service
fairness, and cluster-head energy risk.

The contribution is deliberately scoped to MAC allocation. Cluster-head
selection follows an exogenous HEART-CH schedule; routing and cluster-head
retraining are outside the learned intervention.

## Current paper baseline

The manuscript-facing implementation is frozen as
`HTA_MAC_PAPER_BASELINE_V1_20260901`. Its executable identity is the checkpoint
with SHA-256:

```text
31dc4bbed0b91ff326066dee24db3d550f6df4a347eaca82c728c4b77103934a
```

The common-simulator 100-node reference result is:

| Policy | Delivery | Stale loss | Fairness | RMST | Packets/J |
|---|---:|---:|---:|---:|---:|
| HTA-MAC V1 | 0.42770 | **0.02476** | **0.94511** | 128.28 | 225.77 |
| Cap-corrected residual-energy heuristic | **0.44591** | 0.04838 | 0.87166 | **149.32** | **242.36** |
| Custom online primal-dual | 0.43020 | **0.01700** | **0.98081** | 131.10 | 232.45 |

HTA-MAC V1 is therefore not an all-metric winner. Relative to the corrected
residual-energy heuristic, it moves the operating point toward lower stale loss
and higher fairness, while giving up delivery, restricted-mean survival, and
packets/J. This measured QoS--lifetime trade-off is the defensible result.

The original preregistered confirmation remains preserved, but a later audit
corrected the energy-proportional comparator's cap handling. The corrected
audit supersedes the earlier energy-proportional ranking. It does not alter the
saved HTA-MAC trajectories.

## Methodology

### Policy

The controller uses an `EquivariantSetBranchingC51` architecture:

- a shared local encoder processes every scheduled member;
- masked permutation-invariant global context couples branch decisions;
- a categorical C51 head estimates per-branch return distributions;
- deterministic budget projection converts branch choices into a feasible slot
  allocation;
- trajectory-order and concavity regularization provide structured auxiliary
  supervision;
- QoS multipliers account for delivery, stale loss, and service fairness;
- scheduled-CH reserve, forecast, uncertainty, distance, and feasibility
  context expose role-conditioned energy risk.

The shared branch construction preserves node-identity equivariance, while
projection enforces the global budget. The policy never changes the externally
supplied cluster-head schedule.

### Evaluation protocol

- 20 independent paired confirmation seeds: 3900--3919
- five target ranks nested within each seed
- 3,000-round horizon
- projection and environment budget of 24 slots
- common topology, traffic, harvesting, and schedule realizations for paired
  policy comparisons
- bootstrap confidence intervals over seed-level paired effects
- Wilcoxon signed-rank tests with Holm correction within declared families
- restricted mean survival time for censored first-node-death outcomes
- ten transfer conditions covering node count, traffic, harvesting, battery,
  field scale, and an external PVGIS irradiance trace

The PVGIS input is a real irradiance trace, but the wireless network and radio
remain simulated. The online primal-dual comparator is a custom non-neural
controller, not a reproduction of PPO-Lagrangian, CPO, or another named paper.

Seeds 3900--3919 have been opened and must never be used for future tuning,
selection, or early stopping.

### Additional evidence

The repository includes cap-corrected comparator auditing; scaling from 50 to
300 nodes in increments of 50; matched architecture and auxiliary-loss
ablations; robustness tests across traffic, harvesting, battery, field scale,
and an external solar trace; and confidence intervals, paired tests, effect
sizes, latency, parameter count, and memory/complexity measurements.

Cross-paper headline percentages are contextual evidence only. Different
simulators, traffic definitions, clustering policies, energy models, and
endpoints do not form a valid numerical leaderboard.

## Important provenance

The checkpoint-producing training summary records a failed legacy curriculum
gate and a failed Step-3 development gate. Later independent evaluations are
preserved, but the training gate is not rewritten as a success. This is an
important limitation and a motivation for the next training version.

## Frozen fallback release

The complete V1 fallback is in
[`releases/HTA_MAC_PAPER_BASELINE_V1_20260901/`](releases/HTA_MAC_PAPER_BASELINE_V1_20260901/).
It contains the exact checkpoint, source snapshot, configs, evidence, PVGIS
trace, manuscript, paper-safe claims, SHA-256 manifest, verification script,
and rollback instructions.

Transfer archive:
[`HTA_MAC_PAPER_BASELINE_V1_20260901.zip`](releases/HTA_MAC_PAPER_BASELINE_V1_20260901.zip)

Archive SHA-256:

```text
4c4e18dfc32b903a01b9519a930426e423714c3cde37749a1e65902566315b4a
```

Verify the expanded release:

```powershell
powershell -ExecutionPolicy Bypass -File releases/HTA_MAC_PAPER_BASELINE_V1_20260901/VERIFY_RELEASE.ps1
```

Detailed interpretation and rollback guidance:

- [`PAPER_CLAIMS.md`](releases/HTA_MAC_PAPER_BASELINE_V1_20260901/PAPER_CLAIMS.md)
- [`ROLLBACK.md`](releases/HTA_MAC_PAPER_BASELINE_V1_20260901/ROLLBACK.md)
- [`HTA_MAC_PRIMARY_PAPER_PERFORMANCE_COMPARISON_20260831.md`](HTA_MAC_PRIMARY_PAPER_PERFORMANCE_COMPARISON_20260831.md)
- [`HTA_MAC_FINAL_20SEED_CONFIRMATION_REPORT_20260815.md`](HTA_MAC_FINAL_20SEED_CONFIRMATION_REPORT_20260815.md)

## Repository layout

```text
agents/       branching C51, equivariant policy, projection, and QoS modules
envs/         scheduled intra-cluster MAC environments and accounting
experiments/  training, audit, confirmation, and scalability entry points
config/       frozen experiment contracts and decision thresholds
validation/   invariance, accounting, contract, and regression tests
outputs/      selected checkpoints and machine-readable evidence
paper/        manuscript source and figure-generation utilities
releases/     immutable paper-facing fallback packages
```

## Verification

Run the validation suite from the repository root:

```powershell
python -B -m pytest validation -q -p no:cacheprovider
```

A fresh experiment must be kept separate from the frozen evidence; differences
should be diagnosed, not overwritten.

## Boundary for the next version

Future delivery/energy-efficiency work must use new V2 configs, run names, and
checkpoints. V1 evidence is immutable. V2 should replace V1 only after a
predeclared held-out gate improves delivery and packets/J without erasing the
fairness/staleness contribution.

## Continue in a new Codex chat

New agents should read [`AGENTS.md`](AGENTS.md) and the dated
[`execution handoff`](HTA_MAC_NEW_CHAT_EXECUTION_HANDOFF_20260901.md) before
changing code. A ready-to-paste prompt is available in
[`START_NEXT_CHAT.md`](START_NEXT_CHAT.md).
