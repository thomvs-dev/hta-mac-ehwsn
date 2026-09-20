# Shared-head fragment integration result

The frozen necessary component gate **passed** on 9,216 episodes: 96 scenario
cells, 32 paired development seeds (9710000--9710031), three policies, eight CPU
workers. Simulation execution took 2.827 seconds, excluding validation/analysis.

| Policy | Timely delivery | Packets / charged J |
|---|---:|---:|
| Whole-packet retry | 12.5576% | 41.4527 |
| Per-fragment selective retry | 13.4927% | 44.5781 |
| Selective repair with block ACK | 17.1211% | 57.8723 |

| Block vs baseline | Relative delivery gain | Relative packets/J gain |
|---|---:|---:|
| Whole | +36.340% [33.499%, 39.197%] | +39.611% [36.695%, 42.565%] |
| Selective | +26.892% [25.463%, 28.367%] | +29.822% [28.319%, 31.379%] |

Brackets are descriptive paired-seed 95% bootstrap intervals. Four-claim
family-adjusted lower bounds are respectively 33.079%, 36.289%, 25.247%, and
28.066%; every bound exceeds the unchanged +1% threshold. Bootstrap seed9791000,
20000 replicates, lower percentile .0125. Scenario/policy pairing stays inside
each independent seed. This is development evidence, not held-out confirmation.

The implementation adds source identities, shared head battery and a total
two-packet relay buffer, paid per-source reports, serialized fixed grant windows,
EDF with rotating source ties, max two grants/source and 24 globally per frame.
All packet and energy conservation assertions held. Sources release custody only
on received complete receipts. Metadata expires at deadline; switching heads
invalidates source bitmap knowledge, with no free cache transfer. Eight new
structural tests cover identity rejection, custody under lost receipts, absorbing
death, paid control scaling, head changes and shared resource/energy invariants.

This is a synthetic dedicated-head topology, **not the full HEART-CH integration**.
It has no mobile geometry, endogenous contention, reliable-control failures or
hardware measurements. Its scenario distribution, energy supply and multiple
sources differ from the earlier single-source studies. The sink's energy is
excluded. Delivery remains too low for a deployment/reliability headline.
Known fragment ARQ/block ACK is not a novel architecture. Do not compare these
numbers directly with V1, earlier calendar experiments or external papers.

Original80% attribution Gate A and original network Gate B remain unchanged;
neither is certified here. No neural training. The failed prefix and last-frame
guard remain stopped. Next research requirement: separately preregister the
full HEART-CH wire/queue/handover integration and reproduce this stronger baseline
there, before proposing an adaptive policy or requesting held-out confirmation.

Use `tools/fragment_study.py` and the public evidence instructions for portable
verification/reproduction. Historical runner scripts retain their original local
dependencies solely for source provenance; do not use them as clean-clone commands.
