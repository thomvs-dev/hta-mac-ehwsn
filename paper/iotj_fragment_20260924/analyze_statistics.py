"""Post hoc paired-seed analysis; never changes published gates or raw evidence.

Run with Python, numpy and scipy. Ratios are formed AFTER summing 96 cells
within each seed. Paired t tests address mean seed-ratio differences, whereas
the original bootstrap addresses ratios of aggregate sums. Both are reported.
"""
from pathlib import Path
import csv, gzip, hashlib, json
import numpy as np
import scipy
from scipy import stats

P = Path(__file__).resolve().parent
E = P.parents[1] / 'evidence/fragment_repair_20260921'

def holm(values):
    values = np.asarray(values)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    return adjusted

def main():
    comparisons = {'fixed': [('block', 'whole'), ('block', 'selective'), ('prefix', 'whole'), ('prefix', 'selective'), ('prefix', 'block')],
                   'guard': [('guard', p) for p in ('whole', 'selective', 'block', 'prefix')],
                   'shared': [('block', 'whole'), ('block', 'selective')]}
    rows, seed_rows, inputs, cohorts = [], [], {}, {}
    for study, pairs in comparisons.items():
        units = {}
        for file in sorted((E / study).glob('seed_*.jsonl.gz')):
            inputs[str(file.relative_to(E))] = hashlib.sha256(file.read_bytes()).hexdigest()
            with gzip.open(file, 'rt') as stream:
                records = [json.loads(line) for line in stream]
            seed = records[0]['seed']
            assert seed not in range(3900, 3920) and seed not in units
            units[seed] = {}
            for policy in sorted({r['policy'] for r in records}):
                selected = [r for r in records if r['policy'] == policy]
                assert len(selected) == 96 and {r['scenario'] for r in selected} == set(range(96))
                assert all(r['seed'] == seed for r in selected)
                d, g, e = [sum(r['metrics'][k] for r in selected) for k in ('delivered', 'generated', 'energy_uj')]
                units[seed][policy] = {'delivery_pp': 100*d/g, 'packets_per_j': d/(e*1e-6)}
                seed_rows.append(dict(study=study, seed=seed, policy=policy, delivered=d, generated=g, energy_uj=e, **units[seed][policy]))
        n = len(units)
        assert n == (32 if study == 'shared' else 64)
        cohorts[study] = dict(n=n, seeds=sorted(units), cells_per_seed=96)
        published = json.loads((E / study / 'results.json').read_text())
        for policy, expected in published['aggregate'].items():
            entries = [r for r in seed_rows if r['study']==study and r['policy']==policy]
            d, g, e = [sum(r[key] for r in entries) for key in ('delivered','generated','energy_uj')]
            assert np.isclose(d/g, expected['delivery_ratio'], rtol=0, atol=1e-12)
            assert np.isclose(d/(e*1e-6), expected['packets_per_j'], rtol=0, atol=1e-10)
        for candidate, baseline in pairs:
            for metric in ('delivery_pp', 'packets_per_j'):
                a = np.array([units[s][candidate][metric] for s in sorted(units)])
                b = np.array([units[s][baseline][metric] for s in sorted(units)])
                diff = a-b
                sd = float(diff.std(ddof=1))
                test = stats.ttest_rel(a, b)
                ci = test.confidence_interval(.95)
                positive, negative = int((diff > 0).sum()), int((diff < 0).sum())
                sign_p = stats.binomtest(positive, positive+negative, .5).pvalue if positive+negative else 1.
                original = published['iterations'][candidate]['comparisons'][baseline]['delivery' if metric == 'delivery_pp' else 'efficiency']
                rows.append(dict(study=study, candidate=candidate, baseline=baseline, metric=metric, n=n, df=n-1,
                    candidate_seed_mean=float(a.mean()), candidate_seed_sd=float(a.std(ddof=1)),
                    baseline_seed_mean=float(b.mean()), baseline_seed_sd=float(b.std(ddof=1)),
                    mean_difference=float(diff.mean()), difference_sd=sd, ci95_low=float(ci.low), ci95_high=float(ci.high),
                    t=float(test.statistic), p_raw=float(test.pvalue), cohen_dz=float(diff.mean()/sd),
                    positive_seeds=positive, negative_seeds=negative, tied_seeds=n-positive-negative,
                    sign_p_raw=float(sign_p), published_ratio_gain=original['relative_gain'],
                    published_ratio_ci95=original['paired_95_interval']))
    assert len(rows) == 22
    assert len({s for c in cohorts.values() for s in c['seeds']}) == 160
    for field, target in [('p_raw', 'p_holm_22'), ('sign_p_raw', 'sign_p_holm_22')]:
        for row, adjusted in zip(rows, holm([r[field] for r in rows])):
            row[target] = float(adjusted)
    for name, data in [('seed_endpoints.csv', seed_rows), ('statistical_tests.csv', rows)]:
        with (P/name).open('w', newline='') as stream:
            writer=csv.DictWriter(stream, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    output=dict(status='post_hoc_supplement_original_gates_unchanged', inference_unit='independent seed; 96 fixed scenario cells retained within seed',
        primary_supplement='two-sided paired t tests of mean seed-level endpoint differences',
        sensitivity='two-sided exact binomial sign test excluding exact ties; different null of equal positive/negative probability',
        multiplicity='Holm across all 22 tests, separately for t-test and sign-test families',
        assumptions='independent seed units; t inference assumes approximately normal seed differences; scenarios are fixed, not randomly sampled deployments',
        software=dict(numpy=np.__version__, scipy=scipy.__version__), cohorts=cohorts, input_sha256=inputs, tests=rows)
    (P/'statistical_analysis.json').write_text(json.dumps(output, indent=2, allow_nan=False))
    print(json.dumps([r for r in rows if r['study']=='shared' or (r['candidate'] in ('prefix','guard') and r['baseline']=='block')], indent=2))

if __name__ == '__main__':
    main()
