# HTA-MAC Phase 2D QoS Training and Colab Handoff

**Prepared:** 4 August 2026  
**Scope:** development-only MAC training on frozen HEART-CH schedules; no CH retraining and no held-out Phase 3 use.

## Outcome

Phase 2D now has a runnable QoS-constrained learning objective and a self-contained Google Colab training package. The package trains three independent optimizer lineages (`2299`, `3299`, `4299`) for 500 episodes each, performs development-only selection, and stops before held-out evaluation.

## Implemented objective

The physical simulator reward and reported physical metrics remain unchanged. Before the frozen C51 reward scale is applied, training adds a non-positive Lagrangian penalty for cumulative target-cluster violations of:

- delivery ratio at least `0.55`;
- stale-drop ratio at most `0.45`;
- Jain queue-service fairness at least `0.95`.

These thresholds were frozen from development seeds `2300-2304`. Held-out seeds `3100-3104` were not used. Multipliers update per environment step, are clipped to `[0, 10]`, persist in checkpoints, and reset only the episode counters between curriculum cases.

## Evidence before authorization

The constrained seed-2299 smoke completed 50/50 episodes, 7,826 environment steps, and all 25 seed/cluster curriculum pairs. It reported:

- status `smoke_pass`;
- no non-finite values;
- no always-sleep collapse;
- no pathological domination among the original physical reward terms;
- greedy mean target packets `1547.92`;
- greedy mean FND-free steps `145.04`;
- greedy mean queue fairness `0.8129823`.

The 50-episode smoke does **not** establish convergence or QoS superiority. Its purpose is to show that the new learning path is finite, active, and structurally safe. The post-training Phase 2D audit passed equivariance, cap/budget feasibility, local argmax agreement, and categorical-support boundary gates.

Validation results:

- source workspace: `81 passed`, `93 warnings`;
- freshly extracted final ZIP: `81 passed`, `93 warnings`;
- notebook JSON and all code cells compile;
- all 169 packaged files match the internal manifest.

## Final training files

- Notebook: `colab/HTA_MAC_Phase2D_QoS_Training_Colab_20260804.ipynb`
- Bundle: `colab/HTA_MAC_Phase2D_QoS_Training_Bundle_20260804.zip`
- Bundle checksum: `0be9a01716800a0f79505aa6ba2089573ef496e20c718ab54a3b38fe33278a48`
- Bundle size: `8,114,825` bytes
- Notebook checksum: `510e553dde6511f01dea165d5b3bed1c8f9ec6b818365507fc5a3e73b5e3ce27`

The ZIP is intentionally scoped: source, configuration, five frozen development schedule caches, frozen upstream HEART-CH assets, the historical reference checkpoint required by validation, and compact smoke evidence. Python caches and unrelated outputs are excluded.

## How to run

1. Open the notebook in Google Colab.
2. Select a GPU runtime.
3. Upload the ZIP to `/content/` (renaming is tolerated if it remains the only matching Phase-2D QoS bundle).
4. Run all cells in order.
5. Leave the frozen seeds and 500-episode count unchanged.

The notebook verifies the ZIP checksum, validates every manifested file, installs dependencies, runs the complete test suite, trains each lineage, audits each checkpoint, backs up passing results to Drive, and writes `DEVELOPMENT_SELECTION.json`. A failed curriculum gate or foundation audit stops the notebook rather than silently selecting a failed run.

## Next decision after Colab

Return the generated `HTA_MAC_Phase2D_QoS_Trained_Results_20260804.zip`. Then run exactly one locked Phase 3 evaluation of the development-selected checkpoint on seeds `3100-3104`. Do not retrain or reselect after viewing held-out results. Report right-censoring with Kaplan-Meier/common-horizon restricted event-free time and retain the lifetime/idle-energy versus QoS/freshness trade-off framing unless held-out evidence supports a stronger statement.
