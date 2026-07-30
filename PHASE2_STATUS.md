# HTA-MAC authoritative Phase 2 status

Status: **PASS** on 2026-07-29.

The former single fixed-cluster checkpoint remains a historical sanity gate but
is no longer the authoritative model. The authoritative checkpoint was trained
for 500 episodes on five development seeds and all five cluster ranks using
frozen per-round HEART-CH schedule replay with global node identity preserved.

## Gate evidence

```text
CURRICULUM_PAIRS=25
MAX_PADDED_BRANCHES=100
EPISODES_COMPLETED=500
FULL_CURRICULUM_SEEN=True
ALWAYS_SLEEP_COLLAPSE=False
REWARD_PATHOLOGICAL_DOMINATION=False
GREEDY_MEAN_PACKETS=827.3200
S8_S1_Q_MAX_ABS_DIFF=0.02627563
CONVERGENCE_PASS=True
PHASE2_CURRICULUM_GATE_PASS=True
```

Locked policy settings:

- shared environment capacity `T=24`;
- HTA internal projection cap `B=8`;
- `n_max=3`;
- idle reward weight 3;
- death reward weight 10;
- one gradient update per four stored environment transitions;
- queue-feasible action and Bellman-target caps;
- scheduled dead-CH masking and absorbing death.

Evidence:

- `outputs/phase2/authoritative_dynamic_budget8_500ep/summary.json`
- `outputs/phase2/authoritative_dynamic_budget8_500ep/episodes.jsonl`
- `outputs/phase2/authoritative_dynamic_budget8_500ep/branching_c51.pt`
- `PHASE2_3_REMEDIATION_REPORT.md`

Checkpoint SHA-256:
`0EF29EFAFF04EC1CB652C84A432A53BD0C41D7C68DC9DECFCADF9C277247C2FF`.

Final validation: **25 passed** with
`python -B -m pytest validation -q -p no:cacheprovider`.