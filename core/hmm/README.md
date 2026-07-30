# Frozen HMM assets

Phase 0 references the trained HEART-CH HMM artifacts without copying or
retraining them. The solar artifact is recorded in `../frozen_assets.yaml`.
A trained thermal artifact is required before the Phase 0 gate can pass.



`thermal_auxiliary_params.npz` freezes the current upstream simulator defaults
so later comparisons can remain parameter-identical. Its adjacent JSON manifest
explicitly records `trained: false`; it is not a substitute for fitted thermal
parameters and must never be cited as one.
