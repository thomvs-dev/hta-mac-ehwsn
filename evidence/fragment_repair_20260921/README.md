# Fragment-repair evidence release (21 September 2026)

Three valid synthetic development studies, 64,512 episodes and 160 distinct
paired seed units. Raw compressed episode rows, configurations, scenarios,
source hashes, analysis, intervals and failed gates are included. No downloaded
dataset or trained checkpoint is required for these component studies.

| Study | Episodes | Seeds | Result |
|---|---:|---:|---|
| fixed | 24,576 | 64 | Block passes; prefix fails |
| guard | 30,720 | 64 | Guard fails |
| shared | 9,216 | 32 | Block passes shared-head screen |

`INVALIDATED_PREDECESSOR.json` preserves the exclusion decision for the earlier
timing-flawed run. Its raw data remains in the research workspace and is not
distributed as valid performance evidence. Failed prefix/guard evidence IS
included. V1 and its failed network gates are unchanged.

Install Python 3.13, NumPy and pytest (the author used NumPy 2.4.4). From repo root:

```powershell
python -B tools/verify_fragment_publication_20260921.py
python -B -m pytest validation/test_fragment_repair_fixed.py validation/test_fragment_repair_guard.py validation/test_shared_fragment.py -q -p no:cacheprovider
```

Re-run any published experiment into a NEW output directory, on eight CPU workers:

```powershell
python -B tools/fragment_study.py run fixed outputs/reproduce_fragment_fixed
python -B tools/fragment_study.py run guard outputs/reproduce_fragment_guard
python -B tools/fragment_study.py run shared outputs/reproduce_fragment_shared
python -B tools/fragment_study.py verify shared outputs/reproduce_fragment_shared
```

These commands intentionally replay OPENED published development seeds for
reproducibility, not tuning or fresh confirmation. Existing output paths are
refused. Compare `aggregate` and `iterations` in new `results.json` with the
corresponding public result; runtime and gzip-container timestamps may differ.
Publication verification recomputes statistics from all raw rows, using 20,000
paired-seed bootstrap replicates and each study's original adjusted gate.
No historical local output directories are needed by these commands.

The original dated run/analyze scripts are retained to match their historical
source freezes. They require unshipped local predecessor outputs/logs. Use the
portable commands above, not those archival scripts, in a clean clone.

The new shared study and old studies have different physical scenario grids.
Do not interpret cross-study raw changes as algorithm gains. The shared study
uses dedicated scripted heads, not mobile HEART-CH, and all studies assume
reliable paid controls. Model-generated evidence does not establish hardware
performance, a novel ARQ method, V1 improvement or journal readiness.

See [shared result](../../reports/SHARED_FRAGMENT_RESULTS_20260921.md),
[fixed result](../../reports/FRAGMENT_REPAIR_FIXED_RESULTS_20260920.md),
[guard result](../../reports/FRAGMENT_REPAIR_GUARD_RESULTS_20260920.md), and the
[optional paper subsection](../../paper/fragment_component_section_20260921.tex).
The original 80% attribution Gate A is not recomputed/passed by these screens;
original Gate B is likewise not certified. No neural training was performed.

Local full-workspace validation: 592 passed, 665 dependency warnings in 48.07s
(`LOCAL_VALIDATION.txt`). A clean clone contains only the selected published
tests; the component command above runs 27 tests. Older untracked experiments
were deliberately not added. The historical publication manifest dated13September
describes that commit's tree; its README bytes are retained here as
`README_20260913_ARCHIVED.md` while the root README advances. No old evidence
manifest or result was rewritten.
