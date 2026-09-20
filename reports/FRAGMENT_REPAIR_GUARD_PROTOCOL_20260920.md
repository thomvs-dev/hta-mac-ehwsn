# Deadline guard: iteration after the fixed-window fragment result

Freeze before new performance results. The complete radio, wire, memory, loss,
energy, expiry and scope contract in FRAGMENT_REPAIR_FIXED_PROTOCOL_20260920.md
applies unchanged. Preserve the invalid variable-window run and the valid fixed-
window run; neither is overwritten. No neural training or routing/head changes.

## Motivation and rule

The fixed-window study found a necessary component pass for conventional block
acknowledgements, while affordable partial blocks did not improve delivery and
lost efficiency relative to that block comparator. A distinct hypothesis is that
some late partial attempts spend energy without a funded path to completion.

Candidate guard uses exactly the prefix planner until the last usable frame
(current frame+1 >= packet birth+TTL). In that last frame, the head grants the
plan only if cached fragments OR planned member fragments OR acknowledged sink
fragments cover the whole datagram, AND planned forwarding OR acknowledged sink
fragments cover the whole datagram. Otherwise it abstains for that packet.
All inputs are from local head storage, the paid source report, and received
sink ACKs. This is not a guarantee about hidden sink state: a lost ACK can make
the check conservative and hurt delivery. The experiment must measure that.
The rule uses no future arrival, channel outcome or expected harvest.

Whole, selective, block and prefix all remain unchanged matched comparators.
No candidate is removed for being strong. The gate requires guard to beat ALL
FOUR by >=1% on both delivery and packets/joule, using the adjusted confidence
lower bounds below. Known block ACK is retained as a baseline, not claimed as
our novel architecture. A guard failure stops this candidate without undoing
the earlier component result.

## Fresh development cohort and statistics

Seeds9700200--9700263, bootstrap9789997,64 paired seed units. Exclude3900--3919
and both earlier fragment cohorts9700000--9700063 and9700100--9700163.
Same96 equally weighted scenarios and24-frame horizon;5policies=30720episodes.
Capacity4000/16000uJ, harvest0/300uJ, TTL2/6, two exogenous head profiles, three
data laws, two ACK-loss probabilities; arrivalsBernoulli0.35. Eight CPU workers.

Bootstrap20000 paired seed draws. Four comparisons x two endpoints =8claims;
Bonferroni one-sided familyalpha0.05 gives lower percentile0.00625. Require every
lower bound >=0.01; zero invariant failures. The changed quantile reflects only
the declared comparison count, not a relaxed effect threshold. Ordinary95%
intervals remain descriptive. This is fresh DEVELOPMENT, not final confirmation.

Also report block's raw performance on the fresh cohort descriptively; do not
claim a second formal confirmation gate for comparisons not included in this
eight-claim family. No favorable subset can rescue a failed guard. Original80%
Gate A and original network Gate B remain unchanged and uncertified here.

Freeze guard source, tests, config, runner, analyzer and this contract. Test the
last-frame abstention with a deliberately insufficient battery derived from the
wire planner, and retain all prior protocol/loss/energy tests. Save every raw
seed/scenario/policy row and checksums; recompute means and confidence bounds.
Any next scientific claim must distinguish this synthetic component from full
HEART-CH/V1 performance and the known RFC8931-related mechanisms.
