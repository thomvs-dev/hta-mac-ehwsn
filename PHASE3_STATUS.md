# HTA-MAC Phase 3 status

Status: **held-out pilot complete; Phase 4 not yet executed** on 2026-07-29.

All seven policies use one interface and the identical frozen-schedule MAC
environment. The authoritative five-seed held-out pilot completed 35/35 runs
with no failures or non-finite metrics.

HTA-MAC had zero observed FND and zero observed HND events in all five schedule
windows (1,633-1,730 rounds). Its Kaplan-Meier medians are therefore **not
reached**, not infinite. At the common 1,633-round restriction, HTA's restricted
FND-free and HND-free means are both 1,633 rounds.

Selected medians +/- IQR:

| Policy | Lifetime status | Throughput | Idle J | Fairness | Packets/J | Stale drops |
|---|---|---:|---:|---:|---:|---:|
| Static equal | KM FND/HND 132/163 | 13,247 +/- 256 | 45.137 +/- 0.511 | 0.9221 +/- 0.0276 | 240.53 +/- 3.07 | 17,973 +/- 1,697 |
| S2A2MAC-adapted | KM FND/HND 201/354 | 22,637 +/- 481 | 46.032 +/- 0.222 | 0.8403 +/- 0.0087 | 381.51 +/- 5.80 | 39,389 +/- 971 |
| HTA-MAC | 0/5 FND and 0/5 HND events | 21,374 +/- 437 | 17.057 +/- 1.167 | 0.7804 +/- 0.0215 | 592.26 +/- 12.31 | 143,335 +/- 1,901 |

Paired restricted-time differences versus S2A2MAC are +1,432 FND-free rounds
and +1,279 HND-free rounds. Both five-pair two-sided Wilcoxon p-values are
0.0625, so this pilot does not establish statistical significance.

The principal unresolved issue is QoS: HTA-MAC's survival/efficiency gain is
accompanied by substantially worse stale-packet drops and lower fairness. That
tradeoff must be accepted explicitly or corrected on development seeds before
the 30-trial Phase 4 campaign.

Evidence:

- `outputs/phase3/heldout_pilot_authoritative_budget8_censor_aware/raw_trials.csv`
- `outputs/phase3/heldout_pilot_authoritative_budget8_censor_aware/summary.json`
- `PHASE2_3_REMEDIATION_REPORT.md`
- `BASELINE_PROVENANCE.md`

Final validation: **25 passed**.