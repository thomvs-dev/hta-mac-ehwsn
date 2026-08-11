# Step 3 DQfD-Style Shield Distillation and QoS-Band Projection

## Outcome

The experiment produced an adaptive development candidate, not publication or confirmation evidence.

- The original neural policy without its shield achieved 0/20 joint QoS pairs.
- DQfD-style large-margin distillation from 11,276 shield interventions improved the unshielded policy to 17/20.
- That unshielded policy over-served demand: mean delivery rose to 0.59943 while mean FND fell to 1147.05, below the frozen 1158.75 floor.
- A minimal two-sided action projection then bounded service rather than only adding slots.
- The final selected band (`lower=0.5555`, `upper=0.5560`) achieved 18/20 joint QoS, 20/20 delivery passes, mean delivery 0.55554, mean fairness 0.84303, and mean FND 1159.25.
- The FND margin is only 0.50 rounds, so the candidate is fragile and requires independent confirmation.

## Methodological basis

The implementation adapts two established ideas:

1. Hester et al., *Deep Q-learning from Demonstrations* (2017): combine value learning with a supervised large-margin objective on demonstrated actions. Here the demonstrations are the actions corrected by the already validated QoS/risk shield.
2. Dalal et al., *Safe Exploration in Continuous Action Spaces* (2018), and Alshiekh et al., *Safe Reinforcement Learning via Shielding* (2018): minimally correct policy actions to satisfy explicit constraints. Here the discrete projection adds service below the delivery trajectory and removes only excess service above a narrow band.

This remains a shielded/hybrid RL method. It must not be described as neural-only constrained RL.

## Frozen parameters

- Development seed: 2400
- Distillation optimizer seed: 5701
- Distillation epochs: 5
- Batch size: 64
- Learning rate: 3e-5
- Large margin: 0.8
- CPU threads: 18
- QoS band: 0.5555 to 0.5560
- CH post-forwarding reserve floor: 0.20
- Horizon: 1200
- Budget: 16

## Failed and superseded attempts

- Distillation alone passed delivery but reduced lifetime excessively.
- The first two-sided band sweep found no global candidate: its closest candidate reached 18/20 QoS and FND 1158.60, missing the frozen floor by 0.15 rounds.
- The final boundary refinement was declared in advance as the last adaptive refinement. All three refinement candidates passed the numerical gate; the frozen selection rule chose 0.5560 because it had the highest mean FND (1159.25) among equal 18/20 QoS candidates.

## Claim boundary and next step

The selected result comes from the same adaptive development seed used to construct and tune the controller. It provides mechanism evidence only. No p-value or confidence interval from this single seed would be meaningful.

Before any publication claim:

1. Freeze a new confirmation contract for the complete distilled-checkpoint plus two-sided projection method.
2. Run a fresh unused development confirmation seed without changing any parameter or metric.
3. Only after confirmation passes, freeze untouched evaluation seeds and compare against energy-proportional and S2A2MAC-adapted baselines using paired confidence intervals and a predeclared paired test.
4. Preserve and report the previous one-sided legacy-gate failure; do not rewrite it as a pass.

## Evidence

- `outputs/phase2/step3_dqfd_shield_distillation_seed5701_cpu18/shield_distillation_report.json`
- `outputs/audits/step3_qos_band_projection_sweep_v1/qos_band_sweep_report.json`
- `outputs/audits/step3_qos_band_projection_refinement_v2/qos_band_refinement_report.json`
- `config/step3_qos_band_projection_selected_v2.json`
- `outputs/audits/STEP3_QOS_BAND_SELECTION_AUDIT_20260811.json`

## Primary sources

- Hester et al. (2017), Deep Q-learning from Demonstrations: https://arxiv.org/abs/1704.03732
- Dalal et al. (2018), Safe Exploration in Continuous Action Spaces: https://arxiv.org/abs/1801.08757
- Alshiekh et al. (2018), Safe Reinforcement Learning via Shielding: https://arxiv.org/abs/1708.08611
