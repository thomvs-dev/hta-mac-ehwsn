# Independent paid-loss replication — 12 September 2026

The four-seed paid-loss screen passed its unchanged ten checks: conservative delivery gain 2.2383%, packets/J gain 1.8513%, with positive adjusted intervals. This motivates independent replication, not another architecture change or training. The twenty fresh replication seeds are 380000–380019; no outcomes from this cohort select parameters or checkpoints.

## Fixed study

Retain `envs/paid_control_mac.py`, `experiments/paid_control_trial.py`, both allocator implementations, all radio/protocol parameters and the loss setting unchanged. Member attempts and aggregate forwarding each have 10% erasure probability; forwarding induces within-cluster correlation. Reports, grants and ACKs are charged and reliable when funded. Missing reports mask queue/energy. Full reserved failed-attempt energy remains the conservative abstract cost model. Fixed-size/keyed exogenous draws prevent alive-set-dependent random-stream shifts.

Twenty independent seeds, two policies per seed, forty trials. Reuse one actual frozen HEART-CH mobile replay across policies within each seed. Request 3,000 frames; cohort births 501–2500; retain all early terminations and undefined cohorts. Do not pad schedules or infer outcomes for unborn packets. The 24 data slots are per cluster; additional control phases are not a validated wall-clock or interference model. No hardware, ACK-loss, burst-channel or sustained-horizon claim follows.

## Unchanged gate

Require complete_frontier versus reserve_only: >=1% conservative delivery improvement; >=1% whole-observed-run packets/J improvement; both paired two-sided 97.5% t-interval lower bounds >0 (Bonferroni family alpha .05 for two primaries); candidate mean fairness >=.92; candidate mean stale upper <= reference mean stale lower; all cohorts defined; paired horizons matched; 20 moving nodes verified; zero checked feasibility/packet/energy violations. Both methods face the same paid protocol and loss model.

The prespecified paired bootstrap uses 10,000 draws, RNG 90611 and 97.5% intervals as sensitivity only. It cannot rescue a failed primary gate. All twenty seeds are retained; no optional stopping or favorable subset. Failure or inconclusiveness stops promotion. Historical Gate A/B and the full-coverage failure remain unchanged; no training authorization follows a pass.

## Execution

Branch `codex/v2-paid-loss-replication-20260912`. Contract `config/paid_loss_replication_20260912.json`; output `outputs/paid_loss_replication_20260912`; one CPU worker with bounded native threads. Runner `experiments/evaluate_paid_loss_replication_20260912.py`; finalizer `tools/finalize_paid_loss_replication_20260912.py`. Finalization produces the report, statistical screen, paid-role/packet arithmetic checks, bootstrap sensitivity and `PIPELINE_COMPLETE.json`. Source hashes and fresh-seed inventory are frozen before execution. Existing results and dirty files are preserved.
