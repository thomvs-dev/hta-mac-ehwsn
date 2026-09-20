# HTA-MAC: energy-harvesting WSN scheduling

Research code for MAC allocation under frozen HEART-CH cluster-head schedules. The repository contains a **frozen learned V1 baseline** and a **separate deterministic service-frontier candidate**. The latest candidate is an analytical scheduler; no new neural training has started.

## Latest measured result — 13 September 2026

Independent replication on **20 fresh seeds (380000–380019), 40 paired-policy trials and 64,194 observed frames** passed all ten prospectively specified checks. Both policies pay report, grant and ACK energy; member data attempts and aggregate forwarding each have 10% erasure probability.

| Metric | Reserve-only comparator | Complete-frontier candidate |
|---|---:|---:|
| Mean generated-cohort delivery bounds | 8.8198–9.1380% | 9.2941–9.6109% |
| Whole-observed-run packets/J | 433.2433 | 441.4588 |
| Mean Jain service fairness | 0.982091 | 0.981412 |

**Conservative delivery improves 1.7077%**: candidate lower bound versus comparator upper bound. The paired absolute difference is **0.1561 percentage points**, with a two-sided Bonferroni-adjusted 97.5% interval of **[0.1130, 0.1991] percentage points**. **Packets/J improves 1.8963%**, an absolute difference of **8.2155 packets/J**, adjusted interval **[7.6057, 8.8253]**. Both differences are positive on all 20 seeds; the prespecified bootstrap agrees. Fairness is slightly lower, but remains above the declared 0.92 threshold.

The comparator is author-constructed reserve-only scheduling in the same revised model. These percentages **do not measure improvement over V1 or an external published algorithm**. Absolute delivery remains low under this offered load.

- [Unrounded per-seed evidence and checks](evidence/paid_loss_replication_20260912/summary.json)
- [Detailed result and replay coverage](reports/PAID_LOSS_REPLICATION_RESULTS_20260912.md)
- [Prospective method and stop conditions](reports/PAID_LOSS_REPLICATION_METHOD_20260912.md)
- [Recent-paper comparison and journal gaps](reports/PUBLICATION_PAPER_COMPARISON_20260913.md)

## Architecture and scope

The [complete-frontier allocator](agents/complete_service_frontier.py) enumerates one canonical minimum-energy feasible allocation for every service total from idle through the maximum feasible total, retaining the reserve-only incumbent. It selects using the unchanged reserve surrogate, including member and cluster-head energy. Dominance holds only over this enumerated candidate family, not all tied allocations or long-run outcomes.

Past-harvest EWMA statistics provide a heuristic reserve forecast, not a calibrated confidence bound. The [paid-control environment](envs/paid_control_mac.py) charges reporting, grants, member transmission, CH reception/aggregation/forwarding, idle listening and ACKs. Missing reports mask member state. Keyed node-indexed traffic and harvest draws preserve paired exogenous inputs. Queues, batteries and absorbing death constrain service.

Actual frozen HEART-CH replay has **20 moving nodes out of 100**. Clustering and routing remain exogenous. In this revised model there are **24 data slots per cluster**, plus control phases; this is different from the original V1 action scope and is not a validated network-wide airtime budget.

## What the result does not establish

Schedules terminate after **1,567–1,636 of 3,000 requested frames** because the upstream episode reaches its death threshold. Delivery bounds cover packets generated during the observed portion of the planned birth window (501–2500); they cannot identify unborn packets or sustained 3,000-frame delivery. Pending packets remain explicit. Early terminations and every seed are retained.

Control and ACK channels are reliable when funded. Data erasures are stationary; burst fading, interference, lost ACKs, real packet framing, wall-clock throughput and hardware performance are unvalidated. There is no persistent relay-buffer protocol. The thermal auxiliary is synthetic, as recorded in the frozen asset manifest. These limitations prevent a state-of-the-art or top-tier-journal readiness claim.

Historical Gate A/B failures and the failed full-coverage gate remain failures. This separately contracted observed-replay pass does not authorize neural training. Opened seeds 3900–3919 remain prohibited for future tuning or selection.

## Reproducibility and verification

From the repository root, with Python, NumPy and SciPy installed:

```powershell
python -B tools/verify_publication_summary_20260913.py
```

This recomputes the published statistical comparison from unrounded compact rows. It does not substitute for checking raw frame-level physics or replay mobility.

Full local validation also requires PyTorch, pytest and the original assets:

```powershell
python -B -m pytest validation -q -p no:cacheprovider
powershell -NoProfile -ExecutionPolicy Bypass -File releases/HTA_MAC_PAPER_BASELINE_V1_20260901/VERIFY_RELEASE.ps1
```

Release preparation passed **250 tests** (665 dependency warnings), verified **38 frozen V1 files**, and independently checked all **68 hashes** in the completed replication pipeline.

The original evaluator is `experiments/evaluate_paid_loss_replication_20260912.py`. It deliberately refuses existing output paths and verifies the archived contract. A full rerun requires the matching `../final_repo` HEART-CH source, checkpoint and solar parameters identified in [core/frozen_assets.yaml](core/frozen_assets.yaml), plus the historical files referenced by the [immutable contract](config/paid_loss_replication_20260912.json). **A Git clone alone is not a complete simulation reproduction package.** Do not remove contract checks to bypass missing assets.

The roughly 283 MB aggregate result, detailed frame records and replay binaries remain in the local experiment archive. This publication includes compact rows and their raw-artifact hashes; those hashes establish identity, not public availability. See [publication inventory](evidence/paid_loss_replication_20260912/PUBLICATION_MANIFEST.json). Independent raw-data availability remains a publication task.

## Frozen V1 baseline

Tag **paper-baseline-v1-20260901**, commit `11ef88c336a18833b8511b30dba857ae8b831086`, preserves the learned EquivariantSetBranchingC51 policy. Its 100-node corrected comparison is:

| Policy | Delivery | Stale loss | Fairness | RMST | Packets/J |
|---|---:|---:|---:|---:|---:|
| HTA-MAC V1 | 0.42770 | 0.02476 | 0.94511 | 128.28 | 225.77 |
| Corrected residual-energy | 0.44591 | 0.04838 | 0.87166 | 149.32 | 242.36 |
| Author-constructed online primal-dual | 0.43020 | 0.01700 | 0.98081 | 131.10 | 232.45 |

V1 demonstrates a fairness/staleness trade-off, not universal superiority. Its checkpoint-producing curriculum and development failures remain recorded. Do not compare the V1 table numerically against the revised paid-loss table as an architecture effect.

The [immutable V1 release](releases/HTA_MAC_PAPER_BASELINE_V1_20260901/) retains its checkpoint, source snapshot, manuscript, evidence, [claim boundaries](releases/HTA_MAC_PAPER_BASELINE_V1_20260901/PAPER_CLAIMS.md) and rollback instructions. Archive SHA-256: `4c4e18dfc32b903a01b9519a930426e423714c3cde37749a1e65902566315b4a`.

## Repository guide

- `agents/`: learned V1 and deterministic allocation implementations.
- `envs/`: MAC environments, packet accounting and paid-control model.
- `experiments/`, `config/`: evaluators and frozen study contracts.
- `validation/`: feasibility, accounting and regression tests.
- `evidence/`: compact publishable statistical evidence.
- `reports/`: methods, outcomes and literature comparison.
- `releases/`: immutable V1 fallback.

Read [AGENTS.md](AGENTS.md) before further experiments. Next research priorities are implementation-faithful matched external baselines, complete-horizon evidence under a new contract, protocol/traffic sensitivity, and an independently accessible raw reproduction package.
