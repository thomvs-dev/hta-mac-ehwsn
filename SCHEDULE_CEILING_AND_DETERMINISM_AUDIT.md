# Frozen Schedule Ceiling and Determinism Audit

Date: 2026-07-30

## Outcome

The approximately 1600-1700-round schedule boundary is **not** a hard-coded
schedule-generator cap. It is the inherited HEART-CH environment terminal
condition. The upstream environment ends an episode when fewer than 10% of
nodes remain alive or when no current CH survives.

A fresh deterministic schema-v2 generation for seed 3100 requested 3000 rounds
and stopped at round 1640 with:

```text
stop_reason=upstream_episode_terminated
termination_cause=alive_fraction_below_death_threshold
terminal_alive_count=9
upstream T_FND=1178
upstream T_HND=1539
```

Continuing the schedule past this point would require changing the inherited
upstream episode contract. That is not an in-scope generator fix and will not
be done silently.

## Stop-reason correction

The previous full-horizon generator ignored the `terminated` value returned by
the upstream environment. On the next loop iteration, the CH selector received
an empty alive mask and the generator reported `frozen_policy_selected_no_ch`.
That label was incorrect. Schema v2 records the upstream terminal event and its
cause directly.

## Determinism defect and correction

HEART-CH uses a NoisyNet head whose evaluation action path calls
`reset_noise()`. The old generator seeded the upstream environment but did not
seed Python, NumPy, and Torch at the start of each schedule. Therefore, a
version-1 schedule cache was shared consistently across policies within a run,
but regenerating a schedule from the same network seed alone was not guaranteed
to produce the same exogenous schedule.

Schema v2 now:

1. calls `set_global_seed(seed)` at the start of schedule generation;
2. includes `schedule_schema_version=2` in metadata;
3. includes the schema version in schedule-cache filenames;
4. records terminal alive count, upstream episode statistics, and termination
   cause.

Two independent schema-v2 generations for seed 3100 produced the same complete
CH/embedding signature:

```text
10a571b3e53165075e61b3ee43a4a3b32d24269fa0900095c0e478283ee16312
```

## Consequence for prior results

The old Phase 3 pilot remains a valid **historical paired pilot** because all
seven policies for a seed consumed the same cached schedule. It is not the
schedule artifact to use for new training or Phase 4, and it cannot be claimed
as exactly seed-regenerable without retaining its version-1 cache.

All new training, baseline tuning, development validation, and Phase 4 runs
must use schema-v2 schedules. The planned authoritative multi-training-seed
budget sweep will therefore supersede the old budget-8 checkpoint as the final
model evidence.

Machine-readable evidence:
`outputs/logs/schedule_ceiling_audit_v2.json`.