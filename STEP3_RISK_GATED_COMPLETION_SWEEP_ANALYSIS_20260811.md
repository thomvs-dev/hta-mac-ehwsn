# Step 3 Risk-Gated Completion Sweep Analysis

**Date:** 11 August 2026  
**Status:** completed; no candidate passed  
**Scope:** no-learning development sweep, not model selection

## Decision

No candidate in the frozen 18-point grid passed the joint QoS gate. Therefore
no new bounded training run is authorized from this sweep.

All candidates passed the mean-FND non-inferiority requirement, but every
candidate remained at 0/20 joint QoS pairs and 0/20 delivery passes.

## CPU execution

- four worker processes;
- four Torch threads per worker;
- 16 configured compute threads on 20 logical processors;
- wall time: 989.53 seconds (16.49 minutes);
- useful worker CPU utilization: 72.09% of total logical capacity;
- sum of candidate wall times: 3,644.19 seconds;
- measured parallel speedup: 3.68x;
- peak resident memory was approximately 378 MB per worker.

The initial ten-worker attempt was stopped during imports because ten separate
PyTorch/SciPy runtimes exceeded the Windows paging-file limit. It completed no
candidate. Four workers avoided paging and provided the fastest safe execution
observed on this machine.

## Best observed candidate

`floor0.10_fill0.75_tol0.05`

| Metric | Baseline trained policy | Best completion candidate |
|---|---:|---:|
| Joint QoS passes | 0/20 | 0/20 |
| Delivery passes | 0/20 | 0/20 |
| Mean delivery ratio | 0.486470 | 0.488174 |
| Mean FND-free rounds | 1170.75 | 1170.15 |
| Executed target packets | 175,571 | 175,853 |

The best candidate added only 542 slots over all 20 trajectories. It recovered
less than 1% of the 53,140-packet service gap identified by the oracle
diagnostic.

## Why the layer failed

The CH reserve gate was not the main blocker:

- slots blocked by the reserve gate: 31;
- slots blocked by the marginal-Q tolerance: 45,851.

With zero negative tolerance, every candidate added zero slots. With a 5%
scale-normalized tolerance, candidates added only 515-542 slots. Changing the
completion fraction or reserve floor therefore had almost no effect because
the learned marginal Q values were substantially below the allowed tolerance.

The failure confirms that the learned Q surface itself encodes conservative
service. A post-projector that remains subordinate to those negative marginal
Q values cannot repair delivery.

## Next engineering decision

Do not train longer and do not enlarge this grid post hoc. The next mechanism
must be preregistered as a distinct candidate family. The technically justified
option is a QoS-deficit controller that overrides negative marginal Q only when:

1. cumulative delivery is below a frozen round-indexed target trajectory;
2. feasible target backlog is available;
3. predicted post-forwarding CH reserve remains above a frozen safety floor;
4. the added action stays within budget 16 and the unchanged CH schedule.

Before training, this controller requires another no-learning sweep with a
small frozen grid and the same 18/20 QoS plus 12-round FND gate. The current
result must remain recorded as `no_completion_candidate_passed_stop_training`;
the threshold must not be relaxed after observing the failure.

## Evidence

- `outputs/audits/STEP3_RISK_GATED_COMPLETION_SWEEP_20260811.json`
- `experiments/sweep_step3_risk_gated_completion.py`
- `outputs/audits/STEP3_DELIVERY_FEASIBILITY_DIAGNOSTIC_20260811.json`

