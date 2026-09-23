# Statistical audit and reviewer assessment

The component result is consistent and practically material within its tested
model. It is not yet a demonstrated advance over complete recent IoT protocols.
For IEEE IoT Journal, the main risks remain limited protocol novelty, absent
matched published baselines, short-horizon component validation, and low absolute
delivery under shared-head stress. More p-values do not resolve those risks.

## Relationship to HEART-CH

The supplied paper, Section IV-B, uses two-sided Welch tests reconstructed from
means, confidence-interval half-widths and n=20 because paired traces were not
retained. It corrects two ablation comparisons within each metric using Holm.
Here raw paired traces exist. We therefore use paired t tests with n=64,64,32,
not unpaired Welch tests and not n=64512. Each independent seed contributes one
endpoint after summing all 96 fixed scenario cells. All 22 comparisons in the
three original families are reported, including unsuccessful variants. Holm
adjustment is applied globally across these 22 tests, a broader family than the
reference paper's within-metric correction. This is a post hoc supplement.

## Methods and interpretation

Delivery endpoint = 100 times summed deliveries / summed generated datagrams.
Efficiency endpoint = summed deliveries / summed charged joules, within a seed.
For differences d, t=mean(d)/(sample_sd(d)/sqrt(n)), df=n-1, and paired Cohen
dz=mean(d)/sample_sd(d). Intervals below are pointwise two-sided 95% t intervals.
Normal-difference inference is approximate. The independent unit is the seed;
scenario cells, frames, packets and policy executions are not independent n.

As sensitivity analysis, a two-sided exact binomial sign test checks equal
probabilities of positive and negative differences, excluding exact ties. Its
22 p-values have their own Holm adjustment. It does not test the same mean null
as the t test. All shared comparisons favor block ACK in 32/32 seeds; raw sign
p=4.656612873e-10 and adjusted p=3.725290298e-9. Prefix and guard have lower
efficiency in 64/64 seeds. Delivery has many ties, retained in the t analysis
and explicitly excluded from the sign test's nonzero sample count.

These seed-mean estimands differ from the paper's ratio-of-pooled-sums estimand.
The original paired bootstrap intervals and 1% gates remain authoritative for
headline relative gains. The new tests do not replace them, certify fresh
confirmation, establish equivalence after non-significance, or establish
superiority over external papers. Large dz reflects consistent seed aggregates,
not effect size across arbitrary deployments. No retrospective power claim is
made. No failed Gate A/B or fragment-development gate has been weakened.

## Complete numerical results

D units are delivery percentage points; E units are packets/J. CI is for the
mean paired difference. Exact machine-readable values, seed endpoints, hashes
and software versions are in statistical_analysis.json and the companion CSVs.

| Study | Candidate / baseline | Endpoint | n | Mean difference [95% CI] | dz | Raw p | Holm p (22) | Sign Holm p | Wins / ties |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| fixed | block / whole | D | 64 | 3.613557 [3.115052, 4.112063] | 1.8107 | 1.0096e-21 | 8.0772e-21 | 2.2551e-17 | 60 / 4 |
| fixed | block / whole | E | 64 | 9.418984 [8.263346, 10.574623] | 2.0359 | 2.9028e-24 | 2.9028e-23 | 9.0483e-14 | 60 / 1 |
| fixed | block / selective | D | 64 | 4.098301 [3.612068, 4.584535] | 2.1054 | 5.1687e-25 | 5.6855e-24 | 2.3852e-18 | 64 / 0 |
| fixed | block / selective | E | 64 | 12.684602 [11.812214, 13.556990] | 3.6320 | 3.3608e-38 | 7.0576e-37 | 2.3852e-18 | 64 / 0 |
| fixed | prefix / whole | D | 64 | 3.599413 [3.108497, 4.090329] | 1.8315 | 5.7831e-22 | 5.2047e-21 | 2.2551e-17 | 60 / 4 |
| fixed | prefix / whole | E | 64 | 7.590222 [6.411035, 8.769409] | 1.6079 | 2.7864e-19 | 1.6718e-18 | 8.1025e-12 | 59 / 0 |
| fixed | prefix / selective | D | 64 | 4.084157 [3.612484, 4.555830] | 2.1629 | 1.2739e-25 | 1.5287e-24 | 2.3852e-18 | 64 / 0 |
| fixed | prefix / selective | E | 64 | 10.855840 [10.012981, 11.698698] | 3.2173 | 3.8871e-35 | 7.3856e-34 | 2.3852e-18 | 64 / 0 |
| fixed | prefix / block | D | 64 | -0.014145 [-0.059371, 0.031082] | -0.0781 | 0.53424 | 1 | 1 | 2 / 59 |
| fixed | prefix / block | E | 64 | -1.828762 [-1.942922, -1.714603] | -4.0015 | 1.0999e-40 | 2.4198e-39 | 2.3852e-18 | 0 / 0 |
| guard | guard / whole | D | 64 | 3.068175 [2.639041, 3.497309] | 1.7859 | 1.9689e-21 | 1.3782e-20 | 3.0358e-18 | 63 / 1 |
| guard | guard / whole | E | 64 | 6.684440 [5.552855, 7.816025] | 1.4756 | 1.3056e-17 | 6.5281e-17 | 5.2171e-14 | 61 / 0 |
| guard | guard / selective | D | 64 | 3.569206 [3.208096, 3.930315] | 2.4689 | 1.0973e-28 | 1.8654e-27 | 2.3852e-18 | 64 / 0 |
| guard | guard / selective | E | 64 | 9.700249 [8.923246, 10.477252] | 3.1185 | 2.3339e-34 | 4.2011e-33 | 2.3852e-18 | 64 / 0 |
| guard | guard / block | D | 64 | -0.000142 [-0.032952, 0.032667] | -0.0011 | 0.99312 | 1 | 1 | 3 / 58 |
| guard | guard / block | E | 64 | -1.778085 [-1.913209, -1.642961] | -3.2870 | 1.1279e-35 | 2.2559e-34 | 2.3852e-18 | 0 / 0 |
| guard | guard / prefix | D | 64 | -0.012788 [-0.031390, 0.005814] | -0.1717 | 0.17437 | 0.69748 | 1 | 0 / 62 |
| guard | guard / prefix | E | 64 | -0.019165 [-0.049805, 0.011475] | -0.1562 | 0.21594 | 0.69748 | 0.16556 | 15 / 44 |
| shared | block / whole | D | 32 | 4.566023 [4.298770, 4.833275] | 6.1598 | 1.9854e-26 | 2.5811e-25 | 3.7253e-09 | 32 / 0 |
| shared | block / whole | E | 32 | 16.389694 [15.456128, 17.323260] | 6.3296 | 8.7172e-27 | 1.2204e-25 | 3.7253e-09 | 32 / 0 |
| shared | block / selective | D | 32 | 3.649497 [3.444408, 3.854585] | 6.4157 | 5.7896e-27 | 8.6844e-26 | 3.7253e-09 | 32 / 0 |
| shared | block / selective | E | 32 | 13.278938 [12.610172, 13.947704] | 7.1588 | 2.0736e-28 | 3.3178e-27 | 3.7253e-09 | 32 / 0 |

## Recent-paper comparison

| Work | Original reported evidence | Supported distinction, not an unmatched victory |
|---|---|---|
| RL-ASL, 2026 preprint | FIT IoT-LAB; RL-ASL vs Orchestra average power 0.561 vs 0.996 mW in simple topology and 0.647 vs 1.008 mW in star topology (Table IV). | Receive-slot activation; physical validation is stronger than ours. Our joint energy/custody recovery component addresses a different decision. |
| Van Leemput et al., IEEE Access 2024 | Cooja plus harvesting model, 21 nodes, ten nine-hour runs per condition; reporting 90.19%, uptime 91.10%, latency627 ms (+32.28%). | Energy-aware schedule/routing adaptation with broader network/harvest coverage. These ratios do not share our workload, denominators or horizon. |
| Scanzio et al., 2024 preprint | PRIL-ML at r=4: analytical estimates83.8 uW,9.231 s; PRIL-M simulated68.6 uW,30.58 s slow-flow delay. | Sleep/latency tradeoff. Our mechanism is executed in simulation, but that is not proof of lower power or delay. |

Our demonstrated superiority is +36.34% delivery/+39.61% packets/J against
matched whole retry and +26.89%/+29.82% against matched selective retry, with
unchanged adjusted-bootstrap decisions. Its distinguishing contribution is
jointly funded fixed windows, locally received feedback, bounded custody and
exact conservation ledgers. Selective recovery and block ACK are established
mechanisms, not new inventions. A credible next scientific step is an
implementation-faithful published baseline in a common stack followed by a
frozen, untouched confirmation cohort, not a cross-paper percentage leaderboard.

Sources checked24September2026:
- https://arxiv.org/html/2604.07533v2 (RL-ASL TablesIII-IV, platform sections)
- https://www.famaey.eu/papers/jnl-vanleemput2024a.pdf (SectionVIII-C, Table5)
- https://arxiv.org/html/2411.12879v1 (SectionIV, TablesI-II)
- https://sci2s.ugr.es/keel/pdf/specific/articulo/0052_001.pdf (Holm1979)
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html
- https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binomtest.html

Reproduce from the repository root:
`python paper/iotj_fragment_20260924/analyze_statistics.py`
`python paper/iotj_fragment_20260924/write_statistical_report.py`
