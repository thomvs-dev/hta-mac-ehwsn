# V1/V2 experiment boundary

The frozen V1 release must not be edited while developing a delivery- and
packets/J-oriented V2 policy.

## Rules

1. Treat every file under this release directory as read-only after manifest
   generation.
2. Put future checkpoints under a new run name containing `v2`.
3. Put future configs and results under new `v2` paths. Never overwrite the V1
   checkpoint, confirmation JSON, cap-corrected audit, or scalability result.
4. Use development seeds distinct from 3900--3919. Those confirmation seeds are
   permanently opened.
5. Evaluate V2 against V1 and the corrected heuristic under identical scenario,
   horizon, budget, and statistical units before changing the manuscript.
6. Keep negative candidates and failed gates; do not replace V1 unless a
   predeclared V2 gate passes.
7. If V2 fails, restore `source_snapshot.zip` into a separate directory and use
   the V1 manuscript/evidence bundle without merging failed V2 code into it.

## Minimum replacement gate for V2

The exact gate should be preregistered before reserved evaluation is opened.
At minimum, V2 should improve delivery and packets/J over V1 without destroying
the fairness/staleness contribution that motivates the paper. Cross-paper
headline values are context only and are not a valid acceptance gate.

