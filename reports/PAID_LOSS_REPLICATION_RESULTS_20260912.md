# Censor-aware mobile HEART-CH evaluation — 12 September 2026

Outcome: **observed_replay_screen_pass_not_training_authorization**.

All twenty fresh seeds 380000–380019 are retained. The fixed birth window is 501–2500, but only births actually generated before replay termination are observed. Delivery bounds apply to these generated packets: delivered/generated through (delivered+pending)/generated. They do not identify outcomes of unborn packets, or full 3,000-frame performance. No inverse-censor weighting or independent-censor assumption is made.

| Seed | Policy | Observed frames | Missing birth-window frames | Generated cohort | Pending cohort | Delivery lower–upper | Packets/J | Fairness |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 380000 | reserve_only | 1601 | 899 | 109984 | 364 | 0.08975–0.09306 | 436.213 | 0.98307 |
| 380001 | reserve_only | 1603 | 897 | 110452 | 336 | 0.08659–0.08963 | 432.100 | 0.98265 |
| 380002 | reserve_only | 1607 | 893 | 110500 | 356 | 0.08864–0.09186 | 430.399 | 0.98362 |
| 380003 | reserve_only | 1613 | 887 | 111119 | 358 | 0.08884–0.09206 | 425.131 | 0.98145 |
| 380004 | reserve_only | 1619 | 881 | 112012 | 358 | 0.09112–0.09431 | 440.927 | 0.97733 |
| 380005 | reserve_only | 1576 | 924 | 107507 | 354 | 0.08572–0.08902 | 428.975 | 0.98380 |
| 380006 | reserve_only | 1624 | 876 | 112391 | 342 | 0.08828–0.09132 | 426.695 | 0.98200 |
| 380007 | reserve_only | 1612 | 888 | 111105 | 366 | 0.08553–0.08883 | 427.928 | 0.98371 |
| 380008 | reserve_only | 1567 | 933 | 106921 | 341 | 0.08730–0.09049 | 434.207 | 0.98569 |
| 380009 | reserve_only | 1613 | 887 | 111240 | 348 | 0.08981–0.09293 | 441.517 | 0.98400 |
| 380010 | reserve_only | 1573 | 927 | 107182 | 354 | 0.08871–0.09201 | 434.589 | 0.98254 |
| 380011 | reserve_only | 1627 | 873 | 113363 | 337 | 0.08845–0.09142 | 437.237 | 0.98046 |
| 380012 | reserve_only | 1612 | 888 | 110726 | 363 | 0.08911–0.09239 | 434.206 | 0.98132 |
| 380013 | reserve_only | 1630 | 870 | 112868 | 337 | 0.09124–0.09423 | 440.278 | 0.97855 |
| 380014 | reserve_only | 1613 | 887 | 111736 | 356 | 0.08883–0.09201 | 436.746 | 0.98408 |
| 380015 | reserve_only | 1636 | 864 | 113493 | 350 | 0.08534–0.08843 | 428.854 | 0.97854 |
| 380016 | reserve_only | 1600 | 900 | 110115 | 361 | 0.08936–0.09264 | 435.396 | 0.98634 |
| 380017 | reserve_only | 1601 | 899 | 110953 | 376 | 0.08882–0.09221 | 434.793 | 0.98109 |
| 380018 | reserve_only | 1584 | 916 | 108221 | 346 | 0.08889–0.09209 | 430.803 | 0.98117 |
| 380019 | reserve_only | 1586 | 914 | 108497 | 328 | 0.08363–0.08666 | 427.871 | 0.98041 |
| 380000 | complete_frontier | 1601 | 899 | 109984 | 364 | 0.09350–0.09681 | 442.786 | 0.98325 |
| 380001 | complete_frontier | 1603 | 897 | 110452 | 330 | 0.09091–0.09390 | 440.390 | 0.98099 |
| 380002 | complete_frontier | 1607 | 893 | 110500 | 352 | 0.09510–0.09828 | 440.123 | 0.98192 |
| 380003 | complete_frontier | 1613 | 887 | 111119 | 358 | 0.09276–0.09598 | 432.503 | 0.98085 |
| 380004 | complete_frontier | 1619 | 881 | 112012 | 358 | 0.09468–0.09787 | 448.362 | 0.97793 |
| 380005 | complete_frontier | 1576 | 924 | 107507 | 345 | 0.09075–0.09396 | 436.962 | 0.98318 |
| 380006 | complete_frontier | 1624 | 876 | 112391 | 340 | 0.09371–0.09673 | 436.006 | 0.97935 |
| 380007 | complete_frontier | 1612 | 888 | 111105 | 367 | 0.08953–0.09283 | 435.221 | 0.98312 |
| 380008 | complete_frontier | 1567 | 933 | 106921 | 342 | 0.09203–0.09523 | 441.712 | 0.98504 |
| 380009 | complete_frontier | 1613 | 887 | 111240 | 348 | 0.09393–0.09706 | 447.775 | 0.98383 |
| 380010 | complete_frontier | 1573 | 927 | 107182 | 354 | 0.09463–0.09794 | 444.680 | 0.98129 |
| 380011 | complete_frontier | 1627 | 873 | 113363 | 337 | 0.09382–0.09680 | 445.158 | 0.97952 |
| 380012 | complete_frontier | 1612 | 888 | 110726 | 363 | 0.09340–0.09668 | 442.593 | 0.97968 |
| 380013 | complete_frontier | 1630 | 870 | 112868 | 337 | 0.09539–0.09838 | 447.661 | 0.97802 |
| 380014 | complete_frontier | 1613 | 887 | 111736 | 356 | 0.09403–0.09722 | 445.283 | 0.98325 |
| 380015 | complete_frontier | 1636 | 864 | 113493 | 350 | 0.09004–0.09312 | 438.528 | 0.97749 |
| 380016 | complete_frontier | 1600 | 900 | 110115 | 361 | 0.09357–0.09684 | 442.811 | 0.98749 |
| 380017 | complete_frontier | 1601 | 899 | 110953 | 366 | 0.09468–0.09798 | 444.793 | 0.98096 |
| 380018 | complete_frontier | 1584 | 916 | 108221 | 346 | 0.09385–0.09705 | 439.866 | 0.97980 |
| 380019 | complete_frontier | 1586 | 914 | 108497 | 328 | 0.08850–0.09152 | 435.963 | 0.98126 |

The frozen primary candidate is complete_frontier versus reserve_only. Only the frozen complete-frontier candidate and reserve-only comparator are evaluated. The new screen requires >=1% gain in conservative delivery and observed packets/J, positive Bonferroni-adjusted 97.5% paired t intervals across the two primary comparisons, mean fairness >=0.92, no worst-case mean stale increase, defined cohorts, matched observed horizons and zero checked feasibility errors. Twenty fresh seeds independently replicate the paid-loss model; this is not full-horizon or external-model confirmation. Both methods pay control and ACK costs. Exogenous traffic/harvest draws use fixed-size node-indexed arrays.

Checks: `{'all_seed_cohorts_defined': True, 'matched_observed_horizons': True, 'feasibility': True, 'mobility': True, 'delivery_gain': True, 'delivery_interval': True, 'packets_per_j_gain': True, 'packets_per_j_interval': True, 'fairness': True, 'stale_worst_case': True}`.

- delivery: relative gain 0.01707744734014205; paired difference 0.0015605410523059679; adjusted interval [0.0011298164466251532, 0.0019912656579867826].
- packets_per_j: relative gain 0.018962793996597505; paired difference 8.215504310017504; adjusted interval [7.605723133821645, 8.825285486213364].

Replay coverage and its upstream termination cause:
- Seed 380000: 1601 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380001: 1603 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380002: 1607 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380003: 1613 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380004: 1619 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380005: 1576 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380006: 1624 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380007: 1612 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380008: 1567 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380009: 1613 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380010: 1573 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380011: 1627 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380012: 1612 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380013: 1630 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380014: 1613 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380015: 1636 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380016: 1600 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380017: 1601 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380018: 1584 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.
- Seed 380019: 1586 frames, 20 moving nodes, upstream_episode_terminated, alive_fraction_below_death_threshold.

Executed 40 trials / 64194 observed frames in 1836.59 seconds, using one CPU process under memory pressure. Source and replay hashes are retained. Tests and verification are recorded in the output directory.

The MAC uses revised pre-funded all-cluster service, fixed traffic and current frozen HEART-CH geometry/heads; it does not train CHs or change routing. One identical replay is reused across policies within each seed. The candidate maximizes the unchanged reserve score over all minimum-energy feasible-total allocations, including idle. It does not predict future cluster heads or geometry. Surrogate-score dominance is not a long-run performance guarantee. This separate model charges scheduled reports, grants and ACKs. Member data attempts and aggregate forwarding each have erasure probability 0.10. Control/ACK channels remain reliable when funded. Forward loss affects the whole received cluster prefix; full reserved radio costs are charged even on failure. No hardware evidence is claimed. Read PAID_LOSS_REPLICATION_METHOD_20260912.md for model changes and limits.

Missing birth frames are a truncation of the planned observation window, not a counted packet loss. Pending generated packets remain explicit. If the MAC itself terminates earlier, that row remains and the matched-horizon check prevents promotion. Policy-dependent generated counts, if present, are shown rather than suppressed. Initial battery transients, informative upstream stopping and heterogeneous observation lengths prevent extrapolation to sustained operation.

The earlier full-coverage gate remains failed; this evaluation has a different prospectively declared estimand and does not repair that gate. On failure stop without retuning, shortening the cohort, dropping seeds, changing thresholds, selecting a favorable ablation or starting training. V1 and historical Gate A/B remain unchanged.

Reproduction into absent paths: `python -B experiments/evaluate_paid_loss_replication_20260912.py`, then `python -B tools/report_paid_loss_replication_20260912.py`.
