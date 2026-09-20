# Fragment repair cycle: results

Completed two prespecified iterations of a new MAC-only fragment mechanism
study. The preceding variable-window run is INVALID for performance claims;
this correction charges both nodes through the fixed scheduled receipt boundary.
Research, exact protocol, cost assumptions and stop rules are in
[FRAGMENT_REPAIR_FIXED_PROTOCOL_20260920.md](FRAGMENT_REPAIR_FIXED_PROTOCOL_20260920.md).
Block ACK and selective fragment recovery are established mechanisms; this
experiment is not an RFC8931 implementation or a novelty claim.

## Results

| Policy | Timely delivery ratio | Packets / charged J | Mean charged energy uJ |
|---|---:|---:|---:|
| whole | 11.188% | 33.8514 | 27473.39 |
| selective | 10.667% | 30.5772 | 28999.37 |
| block | 14.458% | 43.2781 | 27769.75 |
| prefix | 14.450% | 41.4350 | 28989.31 |

### block: PASS_necessary_component_screen

- vs whole: delivery +29.226% (95% interval +24.897 to +33.740%; adjusted lower +23.529%); packets/J +27.847% (95% interval +23.632 to +32.264%; adjusted lower +22.319%).
- vs selective: delivery +35.536% (95% interval +32.680 to +38.364%; adjusted lower +31.812%); packets/J +41.537% (95% interval +38.600 to +44.440%; adjusted lower +37.687%).
### prefix: STOP_candidate_gate_failed

- vs whole: delivery +29.156% (95% interval +24.815 to +33.693%; adjusted lower +23.451%); packets/J +22.403% (95% interval +18.312 to +26.688%; adjusted lower +17.007%).
- vs selective: delivery +35.463% (95% interval +32.668 to +38.229%; adjusted lower +31.776%); packets/J +35.510% (95% interval +32.704 to +38.295%; adjusted lower +31.793%).
- vs block: delivery -0.054% (95% interval -0.268 to +0.137%; adjusted lower -0.332%); packets/J -4.259% (95% interval -4.520 to -4.006%; adjusted lower -4.597%).

Gate: BOTH family-adjusted lower bounds must be >=+1% against EVERY required
comparator. Block compares to whole/selective; prefix additionally compares to
block. Neither favorable individual metrics nor scenario subsets rescue failure.
Only component screening is possible here; original80% Gate A, original Gate B,
full-network performance and journal readiness are not certified.

## What was implemented

Paid serialized data fragments, cumulative bitmaps, grants and source reports;
two-hop persistent reassembly; absorbing death, queue/buffer cap2; end-to-end
custody confirmation; deterministic energy/airtime admission. Iteration1 replaces
per-fragment ACK with one ACK per hop block. Iteration2 reduces an unaffordable
block while preserving cached forwarding before new intake. The selective-repeat
baseline ALSO admits affordable partial windows. No future channel/harvest input.

The head cache is released based on the head's received sink confirmation,
not knowledge of whether the source got its final receipt. Head changes invalidate
source knowledge; fragments are not transferred to a new head for free.

## Execution and validation

- 24,576 episodes,96 scenario cells,64 paired development seed units,
 8 workers,2.137s. Seeds9700100--9700163; bootstrap9789998.
- 574 tests passed,665 warnings. All raw packet/energy ledgers
 and saved seed means recomputed; source/output hashes verified. Zero rollout
 invariant failures. V1's38 frozen payloads verified.
- Paired bootstrap20000 repetitions over SEEDS, retaining scenario and policy
 pairing; no treatment of24576 episodes as independent samples. Family10claims,
 lower percentile0.005; bootstrap coverage is approximate. Ordinary95% intervals
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

Branch codex/v2-fragment-repair-20260920. Core: core/fragment_repair_fixed.py; config:
config/fragment_repair_fixed_20260920.json. Run tools/run_fragment_repair_fixed_20260920.py,
then full `python -B -m pytest validation -q -p no:cacheprovider`, then
tools/analyze_fragment_repair_fixed_20260920.py. Scripts refuse existing evidence paths.
Use a new frozen directory for any new study; do not overwrite this run.

Machine evidence: outputs/fragment_repair_fixed_20260920/results.json, comparison.csv,
seed_*.jsonl.gz, SEED_SUMMARIES.json, seed_means.json, SCENARIOS.json, SEED_AUDIT.json,
SOURCE_FREEZE.json, MANIFEST.json, TRACES.json. Logs:
tmp/fragment_repair_fixed_20260920.log and tmp/fragment_fixed_validation_20260920.log.
