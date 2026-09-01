# HTA-MAC new-chat execution handoff

**Prepared:** 1 September 2026  
**Repository:** `F:\WSN\matlab\stage2\hta-mac`  
**Remote:** `https://github.com/thomvs-dev/hta-mac-ehwsn.git`  
**Default branch:** `main`  
**Current pushed commit:** `11ef88c336a18833b8511b30dba857ae8b831086`  
**Frozen tag:** `paper-baseline-v1-20260901`

## 1. What has been completed

The current manuscript-facing HTA-MAC implementation has been frozen as a
self-contained, checksum-verified fallback release. It includes the exact
checkpoint, source snapshot, configs, PVGIS trace, original confirmation,
cap-corrected audit, scalability and publication-extension evidence, manuscript
package, reports, and rollback instructions.

The transfer archive is:

`releases/HTA_MAC_PAPER_BASELINE_V1_20260901.zip`

Its SHA-256 is:

`4c4e18dfc32b903a01b9519a930426e423714c3cde37749a1e65902566315b4a`

The extracted release verifier and an independent extraction/restore test both
passed. The full repository validation suite passed: `174 passed`.

The root README was rewritten around the current corrected evidence and pushed
to GitHub. GitHub's default branch is now `main` at the commit above.

## 2. Exact V1 model identity

- Architecture: `EquivariantSetBranchingC51`
- Learned intervention: intra-cluster MAC slot allocation only
- Exogenous component: HEART-CH cluster-head schedule
- Checkpoint:
  `outputs/phase2/step3_primary_idle_hybrid_100ep_opt6801_cpu6/branching_c51.pt`
- Checkpoint SHA-256:
  `31dc4bbed0b91ff326066dee24db3d550f6df4a347eaca82c728c4b77103934a`
- Evaluation horizon: 3,000 rounds
- Budget: 24
- Opened confirmation seeds: 3900--3919

The source checkpoint's training summary records
`phase2_curriculum_gate_pass: false` and a Step-3 development gate failure.
Later independent evaluations exist, but the failed training gate must remain
visible.

## 3. Current paper-facing result

Corrected 100-node common-simulator means:

| Policy | Delivery | Stale loss | Fairness | RMST | Packets/J |
|---|---:|---:|---:|---:|---:|
| HTA-MAC V1 | 0.42770 | 0.02476 | 0.94511 | 128.28 | 225.77 |
| Corrected residual-energy heuristic | 0.44591 | 0.04838 | 0.87166 | 149.32 | 242.36 |
| Author-constructed online primal-dual | 0.43020 | 0.01700 | 0.98081 | 131.10 | 232.45 |

V1 improves fairness and stale loss over the corrected energy heuristic but
loses delivery, lifetime, and packets/J. The cap-corrected audit supersedes the
earlier uncapped energy-proportional conclusion. Do not claim V1 is an all-
metric winner.

## 4. Why the next version is needed

The current policy is under-serving useful queued packets and/or choosing a
less efficient mix of member service and CH forwarding opportunities. Earlier
workload-conditioned adapters did not improve fresh-seed delivery and packets/J.
Another ungated reward sweep or longer C51 fine-tune is not justified.

The proposed V2 direction is a constrained marginal-utility scheduler:

1. explicitly estimate marginal delivered packets, stale avoidance, fairness
   benefit, member-TX joules, CH receive/aggregate/forward joules, and depleted-
   CH risk for each possible next slot;
2. construct a deterministic MaxWeight/drift-plus-penalty teacher;
3. learn only a permutation-equivariant residual correction;
4. fine-tune with explicit constraints only after the deterministic frontier is
   proven feasible.

The full proposal and predeclared thresholds are in
`HTA_MAC_DELIVERY_EFFICIENCY_IMPROVEMENT_PLAN_20260901.md`.

## 5. Next task: Phase A only

Do not start GPU training. The next agent should:

1. verify the V1 release and inspect `git status` without cleaning user files;
2. create a new branch such as `codex/v2-marginal-utility-audit`;
3. inventory the existing per-step packet and energy accounting in the
   transfer simulator;
4. implement a CPU-parallel mechanism audit for HTA-MAC V1, the corrected
   residual-energy heuristic, and online primal-dual;
5. log, per feasible/allocated slot, real-queue service, delivery success,
   unused feasible demand, unused budget, member-TX energy, CH receive,
   aggregation and forwarding energy, stale avoidance, marginal packets/J,
   member energy, and scheduled-CH energy;
6. use a fresh development cohort, explicitly checking that it excludes every
   previously opened or manuscript-used seed;
7. produce a machine-readable result and a short report decomposing the
   delivery and packets/J gap;
8. stop unless at least 80% of the gap is attributable to observable selection,
   under-service, or marginal-energy mechanisms.

Only after Gate A passes should the deterministic teacher in Phase B be built.
Only after Gates A--C pass should GPU training be considered.

## 6. V2 targets and stop conditions

Primary development targets:

- delivery `>= 0.450`;
- packets/J `>= 244.8`;
- stale loss `<= 0.035`;
- Jain fairness `>= 0.92`;
- RMST `>= 128` rounds;
- zero budget, queue-cap, or dead-node feasibility violations.

Gate B is decisive: if the full-information deterministic teacher cannot reach
these targets, report the structural limit and retain V1. Do not spend GPU
hours hoping a student will exceed an unreachable teacher.

For architecture Gate C, require random-permutation inverse agreement at least
0.99, zero feasibility failures, a same-platform frozen Q tolerance, and unseen-
state teacher top-choice agreement at least 0.95.

For the eventual paper, both delivery and packets/J improvements over the
corrected energy heuristic must have paired 95% confidence intervals entirely
above zero on a fresh confirmation cohort. A recognized constrained-RL baseline
must also be evaluated under matched conditions.

## 7. Essential files

- `AGENTS.md` -- repository rules for every subsequent agent
- `README.md` -- current public project summary
- `HTA_MAC_DELIVERY_EFFICIENCY_IMPROVEMENT_PLAN_20260901.md` -- V2 plan
- `releases/HTA_MAC_PAPER_BASELINE_V1_20260901/README.md` -- frozen identity
- `releases/HTA_MAC_PAPER_BASELINE_V1_20260901/PAPER_CLAIMS.md` -- claim limits
- `releases/HTA_MAC_PAPER_BASELINE_V1_20260901/ROLLBACK.md` -- safe restore
- `config/step4_final_confirmation_v1.json` -- original frozen confirmation
- `outputs/phase4/final_confirmation_v1/final_confirmation.json` -- result
- `config/step4_cap_corrected_energy_audit_v1.json` -- corrected audit contract
- `outputs/phase4/cap_corrected_energy_audit_v1/results.json` -- corrected result
- `config/step4_node_scalability_50_300_v1.json` -- scalability contract
- `outputs/phase4/node_scalability_50_300_v1/results.json` -- scalability result
- `HTA_MAC_Overleaf_Package/main.tex` -- combined manuscript and bibliography

## 8. Known local-worktree warning

The local worktree contains user-owned untracked training bundles, outputs,
reports, and a modified `paper/refs.bib`. Do not clean, reset, move, or commit
them wholesale. The V1 release and pushed commit are clean provenance anchors;
the surrounding working tree is not.

## 9. Ready-to-paste prompt for the new chat

```text
Work in F:\WSN\matlab\stage2\hta-mac. Read AGENTS.md and
HTA_MAC_NEW_CHAT_EXECUTION_HANDOFF_20260901.md completely before acting. Verify
the frozen V1 release, inspect the dirty working tree without cleaning it, and
then begin only Phase A of HTA_MAC_DELIVERY_EFFICIENCY_IMPROVEMENT_PLAN_20260901.md.
Create a new codex/v2-* branch. Implement and run the CPU-parallel marginal
service/energy mechanism audit with fresh development seeds, preserving the
MAC-only HEART-CH scope and excluding opened seeds 3900--3919. Do not start
neural training. Produce machine-readable evidence and a report, enforce the
80% attribution Gate A exactly, and stop/report if it fails. Never overwrite
the frozen V1 release or weaken a failed gate.
```

