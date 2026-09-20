# Fragment repair: research, plan, and pre-run contract

20 September2026. New MAC-only study. All previous source/result freezes and
failed gates stay intact. This is not a numerical continuation of the six-frame
calendar simulator, because a fragment-level channel and transfer model is now
required. It is not a V1 performance improvement claim.

## Research basis and novelty boundary

[RFC8931](https://www.rfc-editor.org/rfc/rfc8931.html) specifies selective fragment
recovery using acknowledgement bitmaps and discusses the cost of losing a whole
datagram after one fragment is lost. Its window guidance explicitly balances
acknowledgement traffic against congestion. Therefore selective repair and block
ACK are established prior art. We test a local energy-feasible variant, not a
new invention of either mechanism or an RFC-conformant reproduction. In
particular, our two-hop fixed MAC, custom wire headers, and absence of IPv6
congestion control differ from the RFC.

[TI CC2420 product data](https://www.ti.com/product/CC2420) lists250kbps,
17.4mA TX and18.8mA RX. At3V, RX is56.4mW. We retain the conservative active
increment of56.34mW above the60uW deep floor. Both battery-powered participants
are charged that envelope through transfer/feedback, even during source waiting
for forwarding. These are declared device-based model costs, not measured mote
energy. Existing timed_sleep_mac.py supplies provenance for startup/guard
assumptions; that frozen file is not modified.

The research-paper workflow was used. Firecrawl Research's known local
ECONNREFUSED127.0.0.1:9 failure was already established in the preceding review;
this bounded follow-up directly verifies primary RFC/manufacturer sources.
No blocked paper download is needed. The previous recent2024--2026 literature
review remains in PRESENCE_RESEARCH_20260920.md. No claim that all papers were read.

## New executable model

One source, two possible exogenous heads, mains-powered sink outside the battery
energy denominator.24 one-second frames. Head1 remains active or changes to2
atframe12. No route/head learning. Each node starts at4000 or16000uJ and harvests
0 or300uJ at frame end, capped at initial capacity. Death at zero is absorbing.
Every living node pays60uJ/frame deep floor; future sleep floors are protected
from data/control admission. No future harvest credit.

At frame start, expire packets, then Bernoulli0.35 arrivals. One arrival maximum
per frame, packet ID equals birth frame. Source cap2 datagrams; each head cap2
live reassemblies. TTL2 or6frames. Forward completion must precede deadline;
unique sink delivery is counted once even if source receipt is lost. Pending,
stale, overflow and source-death losses are separate. Sink receipt state and
delivered IDs cannot be used as free source observations.

Each500-byte packet uses six fragments:5x99 plus5payload bytes. Each has an
8-byte serialized datagram/source/index/count header, plus26bytes PHY/MAC/security
allowance. Largest on-air frame is133bytes (127-byte PSDU+6PHY); all500bytes are
forwarded in successful complete delivery. ACK payload9bytes identifies datagram,
source, head and6-bit cumulative mask. Per-datagram grant14bytes encodes epoch,
identities and selected member/forward masks. Every wire frame has two400us
guards and192us turnaround/gap. Startup1192us and10uJ allowance apply each active
control window. RX-envelope charging includes all failed transmissions/timeouts.

Paid reliable frame sync and source report precede at most two EDF packet grants.
The31-byte source report includes current free energy, queue/deadlines and the
source's last RECEIVED head bitmap. Head uses its local cached fragments and
received sink bitmap. Grants reserve the worst-case funded member, forward and
feedback transaction from both CURRENT batteries; unused forwarding time remains charged to both nodes until the fixed receipt
boundary, regardless of receipt success or hidden fragment outcomes. Energy settlement is deterministic
from the planned/public events, so subsequent source credit subtracts those
debits. Source/head processing2/22uJ per granted datagram also applies.
All event durations must fit1s. At most six member and six forward fragments
per granted datagram, once each per frame; no hidden same-frame retries.

The source retains the full datagram until a paid FULL sink receipt is relayed
successfully. Head may evict payload on its own received FULL sink ACK and retain
a short-lived confirmation record; this does not assume receipt by source.
Source head knowledge resets only on paid discovery of a changed head. Old-head
fragments do not migrate for free; the new head must receive them. Sink partial
reassembly persists until expiry. Partial blocks cannot bypass queue/death limits.

Three data laws: clean; independent fragment/hop success0.9; and hop/frame burst
success0.9 shared by all datagrams/fragments in that frame. ACK/receipt losses
are independent with probability0 or0.1. All failures are charged. Sync/report/
grant are reliable paid controls in THIS study; earlier control-loss results are
not superseded. No interference, collision or security performance claim.

## Baselines and two frozen iterations

- whole: full datagram retransmission on incomplete delivery; no partial
 receiver progress reused for its next attempt. Full block funding required.
- selective: persistent fragments, paid cumulative ACK after each fragment;
 truncates to a jointly affordable prefix. This is the strong selective-repeat
 baseline, not an intentionally all-or-nothing energy admission baseline.
- Iteration1 block: persistent fragments and one cumulative ACK per nonempty
 hop block. Requires all currently missing/planned fragments to be affordable.
- Iteration2 prefix: same block ACK, but when the full window is unaffordable,
 removes new member fragments from its end before removing already cached
 forwarding fragments. Keeps the largest feasible prefix reached by this rule.
 It has no channel oracle or future arrival/harvest access.

All algorithms share the same wire formats, envelope, wake/control schedule,
memory bounds, exogenous channel arrays, traffic and harvesting. They are locally
implemented canonical mechanisms, not claimed reproductions of external papers.
The iteration2 rule is frozen BEFORE either outcome; no post-result threshold
search. It must beat the block comparator as well as the other two baselines.

## Cohort, statistics, gates, stop

Full factorial96 scenarios:2capacities x2harvest x2head profiles x2TTL x3data laws
x2ACK-loss profiles. Equal scenario weights.64 new development seeds9700100--
9700163, checked against registered repository code/config/report seed references
before adding this protocol. Explicitly exclude3900--3919. Eight CPU workers,
one seed per job, all scenarios/policies paired within that seed. The resulting
24576 episodes are NOT24576 independent statistical replicates: there are64
paired seed units. Policies use identical exogenous arrays indexed by time,
datagram, hop and fragment; random draws are not consumed by policy decisions.

Primary endpoints: mean timely unique delivered datagrams and ratio of mean
deliveries to mean charged joules, averaged equally over scenario cells and seeds.
Average within seed over all96 cells, then bootstrap paired SEED units20000times,
resampling all policies together (bootstrap seed9789998). Five comparisons x two
metrics =10 claims. Lower0.005 percentile is the Bonferroni-adjusted one-sided
bound for family alpha0.05; percentile bootstrap coverage is approximate.

The necessary screen requires BOTH lower bounds >=+1% for every comparator,
and zero invariant violations. Block compares with whole/selective; prefix with
whole/selective/block. This strengthens rather than weakens the prior1% rule.
Each failed candidate stops. No subgroup substitutes for the aggregate. The
original80% attribution Gate A and network Gate B are unchanged and NOT passed
by this study. No neural training. A component pass still requires independent
implementation validation, full-network accounting and fresh confirmation.

Freeze core/tests/config/protocol/runner/analyzer before execution. Retain raw
per-seed/scenario/policy rows, source/output hashes, energy categories, bootstrap
details, gate results and reports. Run the complete validation suite.

## Pre-comparison timing correction

The original variable-window implementation was found to undercharge ambiguous
source waits after loss. Its completed run is preserved and INVALIDATED before
using any effect estimates. This separate corrected source/output uses new
seeds9700100--9700163 and fixed windows. Both source and head remain under the
RX envelope until the prepaid end, including idle padding. The final receipt is
scheduled at that fixed boundary. Added regression checks that complete member
loss plus final receipt loss does not reduce either reserved energy charge.
No gate, radio rate, arrival law, scenario weight, or candidate rule was changed.
