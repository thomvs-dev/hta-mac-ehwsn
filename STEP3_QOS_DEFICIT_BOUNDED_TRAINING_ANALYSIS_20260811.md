# Step 3 QoS-Deficit Bounded Training Analysis

## Outcome

The 100-episode CPU run completed in 63.85 minutes. Corrected useful CPU
utilization averaged 85.721% and peaked at 87.939% across 381 valid samples.
One impossible exit-race sample was excluded in a separate corrected summary;
the raw monitor file remains unchanged.

The independent bounded checkpoint gate passed. The legacy full Step 3 trainer
gate did not pass because its active Lagrangian-penalty requirement was zero.
Both results are retained; the bounded pass does not overwrite the legacy
failure.

## Controller-on trajectory

| Episode | Joint QoS | Mean FND | Throughput | Fairness | Packets/J |
|---:|---:|---:|---:|---:|---:|
| 25 | 17/20 | 1164.90 | 98,378.70 | 0.744957 | 2,064.99 |
| 50 | 18/20 | 1162.30 | 98,175.20 | 0.744572 | 2,064.99 |
| 75 | 18/20 | 1161.35 | 98,000.95 | 0.745817 | 2,065.06 |
| 100 | 18/20 | 1161.05 | 97,987.05 | 0.745662 | 2,065.08 |

The last three snapshots pass target QoS. Final mean FND is 9.70 rounds below
the controller-free baseline but remains within the frozen 12-round
non-inferiority margin.

## Attribution

When the controller is disabled on the trained checkpoint, the neural policy
passes 0/20 QoS pairs and has mean delivery 0.487949. With the controller, the
final checkpoint passes 18/20. The learned policy alone therefore did not
acquire delivery feasibility; the deterministic CH-safe action shield provides
the QoS result.

The method must be described as shielded/hybrid RL, not as a neural policy-only
constrained-RL result. Across training and greedy evaluations, the controller
added 181,426 slots and blocked 13,903 additions through its CH reserve gate.

## Required methodological change

The old penalty-activation gate is incompatible with a controller that prevents
delivery violation before the environment step. It is not retroactively waived:
the current legacy gate remains failed. A new controller-aware evidence audit
records shield dependence explicitly and authorizes only a fresh confirmation
on unused development seed 2401 with optimizer seed 5699. Held-out seeds remain
prohibited until that confirmation passes its frozen controller-aware gates.

## Evidence

- `outputs/phase2/step3_qos_deficit_bounded_100ep_seed5599_cpu18/summary.json`
- `outputs/local_cpu_export/step3_qos_deficit_bounded_100ep_seed5599_cpu18/STEP3_BOUNDED_CHECKPOINT_GATE.json`
- `outputs/audits/STEP3_QOS_DEFICIT_TRAINED_CONTROLLER_OFF_DIAGNOSTIC_20260811.json`
- `outputs/local_cpu_export/step3_qos_deficit_bounded_100ep_seed5599_cpu18/cpu_utilization_corrected.json`
- `validation/analyze_step3_qos_deficit_checkpoint.py`
- `config/step3_qos_deficit_confirmation_contract_v1.json`

