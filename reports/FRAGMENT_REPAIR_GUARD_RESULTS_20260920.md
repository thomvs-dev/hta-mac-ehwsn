# Fragment repair cycle: results

Completed the deadline-guard iteration of a new MAC-only fragment mechanism
study. The preceding variable-window run is INVALID for performance claims;
this correction charges both nodes through the fixed scheduled receipt boundary.
Research, exact protocol, cost assumptions and stop rules are in
[FRAGMENT_REPAIR_GUARD_PROTOCOL_20260920.md](FRAGMENT_REPAIR_GUARD_PROTOCOL_20260920.md).
Block ACK and selective fragment recovery are established mechanisms; this
experiment is not an RFC8931 implementation or a novelty claim.

## Results

| Policy | Timely delivery ratio | Packets / charged J | Mean charged energy uJ |
|---|---:|---:|---:|
| whole | 10.885% | 33.0716 | 27461.69 |
| selective | 10.448% | 30.0541 | 29005.87 |
| block | 13.825% | 41.5334 | 27772.39 |
| prefix | 13.821% | 39.7737 | 28993.00 |
| guard | 13.813% | 39.7554 | 28989.93 |

### guard: STOP_candidate_gate_failed

- vs whole: delivery +26.900% (95% interval +22.885 to +31.264%; adjusted lower +21.849%); packets/J +20.210% (95% interval +16.418 to +24.349%; adjusted lower +15.434%).
- vs selective: delivery +32.207% (95% interval +29.452 to +35.057%; adjusted lower +28.690%); packets/J +32.280% (95% interval +29.530 to +35.131%; adjusted lower +28.754%).
- vs block: delivery -0.085% (95% interval -0.337 to +0.114%; adjusted lower -0.426%); packets/J -4.281% (95% interval -4.587 to -4.000%; adjusted lower -4.675%).
- vs prefix: delivery -0.056% (95% interval -0.143 to +0.000%; adjusted lower -0.173%); packets/J -0.046% (95% interval -0.130 to +0.012%; adjusted lower -0.159%).

Gate: BOTH family-adjusted lower bounds must be >=+1% against EVERY required
comparator. Guard compares to whole/selective/block/prefix. Neither favorable individual metrics nor scenario subsets rescue failure.
Only component screening is possible here; original80% Gate A, original Gate B,
full-network performance and journal readiness are not certified.

## What was implemented

Paid serialized data fragments, cumulative bitmaps, grants and source reports;
two-hop persistent reassembly; absorbing death, queue/buffer cap2; end-to-end
custody confirmation; deterministic energy/airtime admission. The new guard uses the affordable block-prefix mechanism but declines a plan
in the final usable frame if its locally known fragments cannot cover a full
delivery. This is conservative under lost sink ACKs and is tested, not assumed safe
for delivery performance. The selective-repeat
baseline ALSO admits affordable partial windows. No future channel/harvest input.

The head cache is released based on the head's received sink confirmation,
not knowledge of whether the source got its final receipt. Head changes invalidate
source knowledge; fragments are not transferred to a new head for free.

## Execution and validation

- 30,720 episodes,96 scenario cells,64 paired development seed units,
 8 workers,2.923s. Seeds9700200--9700263; bootstrap9789997.
- 584 tests passed,665 warnings. All raw packet/energy ledgers
 and saved seed means recomputed; source/output hashes verified. Zero rollout
 invariant failures. V1's38 frozen payloads verified.
- Paired bootstrap20000 repetitions over SEEDS, retaining scenario and policy
 pairing; no treatment of30720 episodes as independent samples. Family8claims,
 lower percentile0.00625; bootstrap coverage is approximate. Ordinary95% intervals
 are descriptive, not substitutes for the adjusted gate.
- Horizon24x1s; capacities4000/16000uJ, harvest0/300uJ/frame, TTL2/6frames,
 arrivalp0.35, six fragments carrying500bytes, at most2 granted packets/frame.
 Data laws clean/independent0.9/hop-frame-burst0.9; ACK loss0/0.1.
- Source config and tests were frozen before performance results. Seed audit
 covers registered code/config/report references and explicitly excludes3900--3919;
 it does not claim to scan all historical binary files.

## Interpretation and stop

These are model measurements using conservative radio envelopes. There is no
hardware validation and no full HEART-CH evaluation. The mains-powered sink is
outside the battery denominator. All comparators receive the same physical
accounting and paid reliable sync/report/grant; data and ACK loss are tested.
The new fragment/error model and24-frame horizon preclude direct numerical
comparison with the earlier calendar aggregates or the100-node V1 results.

Stop every failed candidate; retain its evidence. A necessary component pass
would still require independent protocol verification and a separately frozen
full-network integration/confirmation study. No neural training was run. Do not
retune these opened seeds or portray known ARQ mechanisms as a new architecture.

## Reproduction and files

Branch codex/v2-fragment-repair-20260920. Core: core/fragment_repair_guard.py; config:
config/fragment_repair_guard_20260920.json. Run tools/run_fragment_repair_guard_20260920.py,
then full `python -B -m pytest validation -q -p no:cacheprovider`, then
tools/analyze_fragment_repair_guard_20260920.py. Scripts refuse existing evidence paths.
Use a new frozen directory for any new study; do not overwrite this run.

Machine evidence: outputs/fragment_repair_guard_20260920/results.json, comparison.csv,
seed_*.jsonl.gz, SEED_SUMMARIES.json, seed_means.json, SCENARIOS.json, SEED_AUDIT.json,
SOURCE_FREEZE.json, MANIFEST.json, TRACES.json. Logs:
tmp/fragment_repair_guard_20260920.log and tmp/fragment_guard_validation_20260920.log.
