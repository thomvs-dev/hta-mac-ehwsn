# Completed fragment-repair research and iteration cycle

Three hypotheses were tested through two valid, separately frozen development
studies. A preceding run with an ambiguous wait-time accounting flaw is preserved
and explicitly INVALIDATED; it supplies no claimed performance results.

## Main result

Conventional selective fragment repair with block acknowledgements passed its
necessary component screen. In the corrected first cohort, relative to whole-
packet retries, it increased timely delivery by29.226% and packets/joule by27.847%.
Against the stronger per-fragment selective-repeat baseline, gains were35.536%
and41.537%. Every family-adjusted lower bound exceeded the unchanged+1% margin.

These large RELATIVE gains accompany low ABSOLUTE delivery in a deliberately
energy-constrained24-second synthetic model: whole11.188%, selective10.667%,
block14.458%. They are not gains over HTA-MAC V1 and not evidence of deployable
reliability. The model imposes paid frequent controls and conservative full-window
radio charges. Report the absolute values alongside percentages.

Two proposed extensions failed:

| Candidate | Reference | Delivery change | Packets/J change | Gate |
|---|---|---:|---:|---|
| Affordable block prefix | Block ACK | -0.054% | -4.259% | FAIL |
| Last-frame completion guard, fresh cohort | Block ACK | -0.085% | -4.281% | FAIL |

The guard also failed against the prefix variant. Preserve and stop both failed
candidates. Do not weaken thresholds or use the opened development cohorts to
retune these rules. On the guard cohort, the unchanged block comparator again
had higher raw aggregate delivery/efficiency than whole/selective, but that is
descriptive replication; the formal eight-claim family tested guard only.

## Research interpretation

[RFC8931](https://www.rfc-editor.org/rfc/rfc8931.html) already describes selective
fragment recovery and acknowledgement windows. Therefore the positive result is
a useful implementation/baseline result, not a novel research breakthrough.
Our custom MAC framing is not an RFC-conformant implementation. TI CC2420 typical
radio figures inform a conservative model; no hardware was measured.

The evidence supports adopting block repair as a STRONGER comparator in a future
MAC study. It does not yet support training a neural policy, replacing the V1
manuscript headline, or claiming high-impact journal readiness. Original80%
attribution Gate A and original network Gate B remain unchanged and unpassed by
these component experiments. Exogenous heads and MAC-only scope were retained.

## Implementation and correction

- core/fragment_repair_fixed.py is the valid shared fragment baseline engine.
- core/fragment_repair_guard.py is the separate failed deadline-guard extension.
- Wire headers/grants/ACKs are serialized and validated; packet, buffer, energy,
  deadline, death and action-cap invariants are asserted in every rollout.
- Both nodes remain charged through the fixed grant window; missing fragments
  or lost receipts cannot silently shorten a source's required wait.
- Payload caches are bounded to two live datagrams per head. Confirmation bitmap
  metadata can remain until head death or horizon end; these are bounded small
  simulation records, not measured firmware RAM or a complete reboot protocol.
- Reliable paid sync/report/grant is assumed here. Fragment, ACK and end receipt
  loss is modeled. Earlier control-loss experiments are not superseded.

## Evidence and next decision

Valid studies:24576 +30720 =55296 episodes on two disjoint64-seed development
cohorts (128distinct seed units total). Eight CPU workers per study. Latest full
validation:584passed,665warnings. Earlier invalid24576episodes are excluded.

Use FRAGMENT_REPAIR_FIXED_RESULTS_20260920.md for the positive component result
and failed prefix result; FRAGMENT_REPAIR_GUARD_RESULTS_20260920.md for the third
iteration. Their corresponding PROTOCOL files and config JSONs were frozen
before their respective outcomes. Raw rows, paired bootstrap intervals and all
comparisons are in outputs/fragment_repair_fixed_20260920 and
outputs/fragment_repair_guard_20260920. Final audit:
outputs/fragment_repair_guard_20260920/FINAL_CYCLE_VERIFICATION.json.

Next justified phase: independently audit/port the valid block-repair baseline
to the larger MAC environment under a newly frozen integration contract. Preserve
its full wire/energy cost, buffer lifetime and head-handover semantics. Reproduce
the baseline there before proposing a new learned or adaptive extension. Any
later architecture must beat this stronger comparator on fresh seeds; the two
failed window rules are not candidates for training. No further experiments or
background jobs are promised by this handoff.

Branch: codex/v2-fragment-repair-20260920. V1's38payloads and tag target
11ef88c336a18833b8511b30dba857ae8b831086 are preserved. Existing tracked changes
to AGENTS.md and paper/refs.bib were not modified by this cycle.
