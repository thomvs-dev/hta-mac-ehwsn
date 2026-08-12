# Final matched-simulator results capsule

## Design

The frozen primary idle-listening-on simulator was evaluated on 20 previously unused seeds (3700-3719), with five target-rank schedules nested within every seed. The independent inferential unit was the seed (n=20), not the 100 correlated seed/rank rows. Each policy used the same schedules, 3,000-round horizon, frozen HEART-CH schedule, and MAC-only action scope. No selection or retuning occurred after the cohort was opened.

The prespecified family contained five metrics against three fixed comparators (15 paired hypotheses). Two-sided Wilcoxon signed-rank tests used seed-level means, 95% paired-bootstrap confidence intervals used 20,000 resamples, and p-values were Holm-corrected across the full family.

## Main result

HTA-MAC passed the joint delivery/staleness/fairness constraints in **99/100** matched trials. The tuned energy-proportional baseline passed 97/100, FFSS-adapted 1/100, and S2A2MAC-adapted 0/100.

Against tuned energy-proportional, HTA-MAC had slightly lower delivery (paired difference -0.0058, 95% CI [-0.0097, -0.0021], Holm p=0.01923) but lower stale ratio (-0.0047, 95% CI [-0.0060, -0.0033], Holm p=0.0001335) and higher service fairness (+0.0504, 95% CI [0.0378, 0.0632], Holm p=4.005e-05).

The lifetime result is a genuine trade-off, not dominance: HTA-MAC reached FND 7.33 rounds earlier than energy-proportional (95% CI [-8.36, -6.24], Holm p=0.0003533) and delivered 0.665 fewer packets/J (Holm p=0.0001144). The defensible contribution is therefore reliable joint-QoS control and learned approximation of the analytic projection, not universal lifetime or efficiency superiority.

## Claim boundary

These are matched-simulator comparisons. S2A2MAC and FFSS are documented structural adaptations because their original simulators and unpublished parameters are not interchangeable with this environment. The results must not be presented as direct reproduction of, or numerical superiority over, the source papers.
