# Next study: completion-aware admission and adaptive feedback

Status: planned, not implemented or evaluated. Prepared 24 September 2026.
This is a new deterministic MAC component study, not a claim of improvement or
authorization to bypass the original HTA-MAC gates. No neural training is planned.

## Starting evidence and objective

The published shared-head block-ACK comparator delivers 17.1211% of offered
datagrams at 57.8723 packets/J. It exceeds matched whole and selective retry,
but does not establish superiority over an external protocol. Prefix and
completion-guard variants failed their frozen gates and remain failed.

The supplementary paired analysis uses n=64,64,32 independent seeds and global
Holm correction across 22 tests. Its very small p-values reflect consistent
differences on fixed scenario grids, not external validity or novelty. Preserve
the original ratio-of-sums bootstrap estimand and distinguish it from mean
seed-ratio t tests. Exact p-values remain in the evidence; emphasize effect sizes
and confidence intervals in claims. Do not change tests to obtain a desired p.

Research question: can admission based on the probability of timely completion,
combined with feedback timing that accounts for uncertainty, deliver more useful
application messages per joule without sacrificing delivery or fair service?

## 1. Freeze the comparator and diagnose the mechanism

Use published component evidence commit 0e12c49 and the current block-ACK
implementation as immutable references. Start implementation on a new
`codex/v2-completion-feedback-*` branch. Use separate config, outputs and reports.
Never modify V1, past evidence, or past gate thresholds.

Before designing a controller, inspect shared-head state transitions and instrument:
- packet age, deadline slack and received missing-fragment bitmaps;
- source/head funded energy and protected background reserve;
- buffer occupancy, admission attempts and repeated rejections;
- complete delivered messages, expiry, overflow, custody retention and receipts;
- data, control, listening, padding and processing energy, separately.

Run paired diagnostic interventions on development inputs: vary head energy
alone, live-buffer capacity alone, and both, retaining the baseline setting.
These are bottleneck diagnostics, not policy improvements. Report their extra
resources explicitly. Couple exogenous events by physical identity. Any
full-information oracle is a labelled upper bound, never a deployable policy.
If timing or accounting defects appear, stop the experiment, preserve its output,
and write a separately versioned repair contract before rerunning.

Decision D0: proceed only if diagnostics show a recoverable loss mechanism under
the ORIGINAL resource budgets. If improvements require more battery or buffers,
report that resource frontier rather than claiming a scheduling breakthrough.
This decision is not the historical 80% attribution Gate A; that gate remains
unchanged and cannot be certified by this component study.

## 2. Implement a bounded completion-aware admission hypothesis

Use only paid observations, received acknowledgements and past channel outcomes.
Never expose hidden reception, future arrivals, future harvest or future erasures.
For packet p and feasible action a, estimate

    U(p,a) = Pr(timely unique completion | received state, a)
             / expected additional charged joules.

Include both hops, feedback, custody, and the finite horizon in the estimate.
Estimate uncertainty and calibration on development data only. A small bounded
enumeration or dynamic program is preferable to a new neural controller here.
The score is a hypothesis, not a proof of optimality. Compare it with EDF and
simple remaining-fragment ranking to test whether complexity adds value.

Admission must reserve the same complete transaction envelope as the comparator,
retain custody until a received receipt, and obey queue, buffer, grant and frame
limits. Do not evict acknowledged payload invisibly. If selective admission
starves a source, constrain or reject the candidate on development evidence.

This differs from the failed prefix/guard rules: it values completion before
consuming scarce shared state, rather than merely admitting an affordable prefix
or applying a last-frame guard. It must still demonstrate that distinction in
ablations; renaming the failed policy is not an innovation.

## 3. Implement adaptive feedback as a separate hypothesis

Compare a bounded set of feedback group sizes, including the fixed block-ACK
incumbent. Select a group using received uncertainty, deadline slack and paid
cost; assess the value of earlier knowledge versus additional ACK cost.

Announce group boundaries before transmission and charge the actual reserved
windows, padding and failed feedback. If a new wire field or control message is
needed, charge its bytes and airtime. Do not shorten listening after hidden
failure. Freeze feasible action choices and tie-breaking before the screen.

Compare four arms: unchanged block ACK, completion-aware admission only,
adaptive feedback only, and their combination. Keep whole/selective retry as
context, not the strongest required gate comparator. Report all arms, including
regressions. If each extension is harmful, stop rather than adding complexity.

## 4. Tests, cohorts and prospective decision rules

Before performance execution, require invariants for zero/complete loss, lost
ACKs/receipts, duplicate fragments, deadline boundaries, exhausted head energy,
full buffers, head changes, simultaneous arrivals and variable source counts.
Check nonnegative energy, absorbing death, unique delivery, packet/energy
conservation, local observability, queue/buffer limits and serialized airtime.
Use the repository validation suite and targeted tests for the new mechanisms.

Inventory every seed ledger before allocating new seed IDs. Explicitly exclude
3900--3919 and all opened development/confirmation seeds, including
9700100--9700163,9700200--9700263,9710000--9710031. Record the full overlap audit,
exact IDs, random-stream rules, config hashes and source commit BEFORE execution.
Do not invent fresh seed IDs in this document without that audit.

Separate engineering/pilot, candidate screen, and untouched confirmation.
Use the pilot only for implementation, calibration and sample-size planning.
Fix candidate definitions, scenario weights, resource budgets and endpoints
before opening screen outcomes. Select at most one candidate for confirmation.

Proposed screen rule (freeze before running): candidate must have simultaneous
one-sided lower bounds of at least +1% on BOTH aggregate timely delivery and
packets/J against unchanged block ACK. There are six efficacy claims for three
new arms; use seed-clustered bootstrap lower quantile 0.05/6 with 20,000 resamples.
Preserve all 96 cells of a seed together. This is a new component screen and does
not replace historical Gates A/B or prior failed fragment gates.

Also freeze service guardrails before the screen: Jain fairness on unique timely
deliveries per source must not decline by more than 0.01, and expiry fraction
must not increase by more than 0.005 (absolute). Define Jain=0 when all sources
have zero delivery, and report that case separately. Apply a separate simultaneous
family to these six safety comparisons using seed-level resampling. Passing
efficacy and both safety requirements is mandatory. These margins are proposed
engineering tolerances, not evidence of application acceptability.

Plan confirmation sample size from pilot paired variability with an explicit
power calculation for an engineering target of +5% in both primary endpoints
against the +1% margin. Fix alpha, target power (90%), resampling method and n
before confirmation. Use at least 32 independent seeds; increase n if justified
by prospective power, not because a completed test failed. If required n exceeds
the compute budget, report the limitation rather than claim adequate power.

Confirmation uses two primary efficacy claims for ONE frozen selected candidate;
use family-adjusted lower bounds at 0.05/2, with safety checks separately frozen.
No optional stopping, checkpoint selection or retuning on confirmation. Report
absolute delivery, energy, expiry, fairness and full intervals regardless of
outcome. Include withheld traffic/harvest/channel conditions declared in advance;
define any additional claims and multiplicity before opening them.

## 5. Establish a fair comparison with published work

Implement and test an appropriate standard selective-recovery mechanism first;
RFC8931 is a relevant starting specification, but do not call a simplified
bitmap comparator an RFC-conformant reproduction. Record any omitted functions.

For an end-to-end claim, use a common Contiki-NG/Cooja configuration with the
published RL-ASL/PRIL-M implementations where licensing and compatibility permit.
First reproduce their stated reference scenarios to detect implementation drift.
Then compare identical application payloads, fragmentation, offered load, losses,
radio costs, topology, head schedule, horizon and seeds. Run a factorial comparison
of listening mechanism and recovery mechanism to distinguish contributions.
This larger integration is conditional on a successful component screen.

Freeze application reliability and latency requirements before its experiments;
do not choose a target because it favors our method. Report energy at comparable
service quality and complete energy--delivery tradeoffs. RL-ASL's hardware results
cannot be replaced by our simulation percentages; no hardware is currently
available. Head selection and routing remain exogenous in our claimed scope.

Sources to use for implementation planning:
- https://www.rfc-editor.org/rfc/rfc8931
- https://arxiv.org/html/2604.07533v2
- https://github.com/fdojurado/contiki-ng-rl-asl
- https://www.famaey.eu/papers/jnl-vanleemput2024a.pdf
- https://arxiv.org/html/2411.12879v1

## Execution order and deliverables

1. Mechanism and observability audit; frozen baseline hashes; seed inventory.
2. Paired bottleneck diagnostics; decision D0; machine-readable causal contrasts.
3. Two bounded mechanisms and four-arm ablations; structural validation.
4. Frozen screen; raw rows, per-seed sums, adjusted intervals, unchanged decision.
5. Conditional matched-stack reproduction and prospective confirmation.

Parallelize independent seeds using a worker count based on available RAM/CPU,
initially eight workers, leaving system headroom. Timing performance must include
analysis separately from rollout execution. No GPU or neural training is needed.

Every run must produce the source/config hashes, seeds, raw episode rows,
per-seed endpoints, invariants, runtime/worker metadata, confidence intervals,
effect sizes, family definitions and explicit pass/stop decision. The final
report must distinguish prediction, simulation, external published measurements,
and proposals. Success is not guaranteed; a failed screen ends that hypothesis.
