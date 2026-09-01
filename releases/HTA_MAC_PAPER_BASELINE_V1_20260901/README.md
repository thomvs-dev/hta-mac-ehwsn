# HTA-MAC Paper Baseline V1

Frozen on 1 September 2026 before any delivery/energy-efficiency redesign.

This release is the recoverable reference for the manuscript-facing HTA-MAC
implementation. It is intentionally independent of the repository's current
Git commit because the working tree contains material tracked and untracked
paper artifacts. The source snapshot, evidence files, manuscript, and hashes
in this release define the baseline.

## Model identity

- Release ID: `HTA_MAC_PAPER_BASELINE_V1_20260901`
- Policy architecture: `equivariant_set_branching`
  (`EquivariantSetBranchingC51`)
- Learned scope: intra-cluster MAC slot allocation only
- Cluster-head schedule: exogenous; it is not learned or modified by HTA-MAC
- Source checkpoint: `branching_c51.pt`
- Checkpoint SHA-256:
  `31dc4bbed0b91ff326066dee24db3d550f6df4a347eaca82c728c4b77103934a`
- Evaluation horizon: 3,000 rounds
- Projection/environment budget: 24
- Primary confirmation units: paired seeds 3900--3919
- Repository branch at freeze: `codex/listwise-residual-ablation`
- Repository HEAD at freeze:
  `c9794d31c3b4e74c9789ed28e2c4ac51a6b8545a`
- Working tree at freeze: dirty; the source archive is authoritative, not HEAD
  by itself.

## Evidence hierarchy

1. `final_confirmation.json` is the original preregistered 20-seed result. It
   retains historical value for the declared comparison with the custom online
   primal-dual controller and for the multi-scenario evaluation.
2. `cap_corrected_energy_results.json` is the later implementation audit of the
   energy-proportional heuristic. It supersedes every conclusion that depended
   on the earlier uncapped energy-proportional implementation.
3. `node_scalability_results.json` and the publication-extension results are
   supplementary analyses. They do not convert the common-simulator evidence
   into cross-paper numerical superiority.

## Current paper-facing reference result

At 100 nodes in the common transfer simulator:

| Policy | Delivery | Stale loss | Fairness | RMST | Packets/J |
|---|---:|---:|---:|---:|---:|
| HTA-MAC V1 | 0.42770 | 0.02476 | 0.94511 | 128.28 | 225.77 |
| Cap-corrected residual-energy heuristic | 0.44591 | 0.04838 | 0.87166 | 149.32 | 242.36 |
| Custom online primal-dual | 0.43020 | 0.01700 | 0.98081 | 131.10 | 232.45 |

Therefore V1 is not an all-metric winner. Its defensible result is a measured
fairness/staleness trade-off: relative to the cap-corrected energy heuristic,
it reduces stale loss and increases fairness, while delivery, RMST, and
packets/J are lower.

## Important qualification

The checkpoint-producing training summary records
`phase2_curriculum_gate_pass: false` and a Step-3 development gate failure,
even though later independent evaluation artifacts exist. Preserve and report
that provenance; do not rewrite it as a successful training gate. The release
exists to make the evaluated policy reproducible and the evidence auditable,
not to erase negative gates.

Seeds 3900--3919 are opened confirmation seeds. Never use them to tune, select,
or early-stop a future model.

## Release contents

- `source_snapshot.zip`: executable repository source at the freeze boundary
- `artifacts/`: exact checkpoint, configs, traces, and result JSON files
- `manuscript/`: the compile-ready manuscript package at the freeze boundary
- `reports/`: the interpretation and evidence reports used for paper writing
- `ARTIFACT_MANIFEST.json`: size and SHA-256 for every frozen payload
- `PAPER_CLAIMS.md`: reviewer-safe statements and forbidden overclaims
- `ROLLBACK.md`: verification and restore procedure
- `EXPERIMENT_BOUNDARY.md`: isolation rules for the next model version

