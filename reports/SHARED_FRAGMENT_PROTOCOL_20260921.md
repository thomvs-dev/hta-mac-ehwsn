# Shared-head fragment integration: frozen development protocol

Freeze this file, code, tests, config and runner before performance execution.
The hypothesis is that the unchanged block-ACK comparator retains a necessary
advantage over whole-packet and per-fragment selective retry when multiple
sources share a head battery, two live reassembly buffers and a one-second frame.
No new mechanism or novelty is claimed.

96 equally weighted scenario cells: 1/4/12 sources, capacity 16000/64000 uJ,
harvest 0/1200 uJ per frame, TTL 2/6, fixed/midpoint-switched scripted head,
independent/burst data loss. Data success .9, ACK/receipt loss .1, arrivals .35.
24 frames; two possible dedicated heads, mains-powered sink excluded from energy.
Sources and heads pay 60 uJ per alive frame, preserve remaining deep-sleep costs,
and cannot spend future harvest. Death is absorbing. Each source queue holds two
packets; each head has two TOTAL live payload reassemblies. All bitmap metadata
expires at the packet deadline. No free head-handover transfer.

One reliable paid broadcast sync precedes individually paid reliable reports.
The head pays every report reception. Grant windows serialize on a common frame;
both participating radios pay the full worst-case duration even on hidden loss.
Other nodes pay deep sleep. Source-specific reports and wire identities prevent
cross-source fragment reuse. Each 500-byte packet traverses both hops in full;
no free compression. EDF with frame-rotating source ties; max 2 grants/source and
24 total. Head cache admission is conservative: when full, new packet identities
are refused even for confirmation-only requests. Dedicated heads generate no data.
Reliable controls and modeled radio costs remain important limitations.

Fresh development seeds 9710000--9710031, 8 CPU processes; exclude 3900--3919 and
all previously opened fragment cohorts. Check registered text references before
running, not an exhaustive audit of arbitrary historic binary files.
Bootstrap seed 9791000; 20000 paired seed resamples retaining all scenario and
policy pairings. For BOTH delivery and packets/J versus BOTH prespecified
baselines, the lower percentile .0125 must be >= +1% (four claims, family .05).
Ordinary 95% intervals are descriptive. Preserve and STOP any failed gate;
no scenario selection, seed expansion, retuning, or threshold changes afterward.

This intermediate integration does not run the full 100-node HEART-CH environment,
mobile geometry or the V1 workload. Original 80% attribution Gate A and Gate B
are unchanged and cannot be passed by this study. No neural training. Publish
absolute delivery as well as relative gains; no cross-study V1 percentage claim.
