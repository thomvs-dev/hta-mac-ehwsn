# HTA-MAC

HTA-MAC is a bounded intra-cluster MAC research project built on a frozen
HEART-CH cluster-head policy. Routing and cluster-head retraining are outside
this repository's scope.

## Current phase

Phase 0 (Foundation) only. Phase 1 code must not be added until every Phase 0
gate check passes.

The authoritative upstream snapshot is the clean HEART-CH V9.1 repository at
`../final_repo`, commit `d96abce25237feb2b6d6c660f6b4d605feb94330`.
The checkpoint and Stage 1 HMM artifacts are referenced in
`core/frozen_assets.yaml`; they are not duplicated here.

## Phase 0 gate

From this directory:

```powershell
python -B validation/phase0_gate.py --upstream ../final_repo
python -B validation/phase0_gate.py --upstream ../final_repo --run-evaluation --episodes 30 --seed 1000
```

The second command evaluates only the frozen HEART-CH policy. It does not run
the comparison baselines and does not train or modify any model.

