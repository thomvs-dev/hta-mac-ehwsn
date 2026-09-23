from pathlib import Path
import json
P=Path(__file__).resolve().parent
r=json.loads((P/'statistical_analysis.json').read_text())
intro='''# Statistical audit and reviewer assessment

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
'''
lines=[]
for x in r['tests']:
    lines.append(f"| {x['study']} | {x['candidate']} / {x['baseline']} | {'D' if x['metric']=='delivery_pp' else 'E'} | {x['n']} | {x['mean_difference']:.6f} [{x['ci95_low']:.6f}, {x['ci95_high']:.6f}] | {x['cohen_dz']:.4f} | {x['p_raw']:.5g} | {x['p_holm_22']:.5g} | {x['sign_p_holm_22']:.5g} | {x['positive_seeds']} / {x['tied_seeds']} |")
end='''

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
'''
(P/'STATISTICAL_REVIEW.md').write_text(intro+'\n'.join(lines)+end)
