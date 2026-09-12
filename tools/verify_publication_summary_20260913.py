"""Recompute the published paired screen from compact, unrounded seed rows."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]


def main():
    evidence = json.loads((ROOT / 'evidence/paid_loss_replication_20260912/summary.json').read_text())
    rows = evidence['rows']
    data = {p: sorted((r for r in rows if r['policy'] == p), key=lambda r: r['seed'])
            for p in ('reserve_only', 'complete_frontier')}
    assert len(rows) == 40
    for group in data.values():
        assert [r['seed'] for r in group] == list(range(380000, 380020))
        for r in group:
            assert r['invariants_pass'] and r['bounds']['defined']
            q = r['cohort']
            assert q['generated'] == sum(q[k] for k in ('delivered', 'stale', 'death', 'overflow', 'pending'))
            np.testing.assert_allclose(r['bounds']['delivery_lower'], q['delivered']/q['generated'])
            np.testing.assert_allclose(r['bounds']['delivery_upper'], (q['delivered']+q['pending'])/q['generated'])
    ref, cand = data['reserve_only'], data['complete_frontier']
    assert all(a['rounds'] == b['rounds'] for a, b in zip(ref, cand))
    assert sum(r['rounds'] for r in rows) == 64194
    for metric, a, b in [
        ('delivery', [r['bounds']['delivery_lower'] for r in cand], [r['bounds']['delivery_upper'] for r in ref]),
        ('packets_per_j', [r['packets_per_j'] for r in cand], [r['packets_per_j'] for r in ref]),
    ]:
        a, b = np.array(a), np.array(b)
        d = a-b
        half = t.ppf(.9875, 19)*d.std(ddof=1)/np.sqrt(20)
        ci = [d.mean()-half, d.mean()+half]
        gain = a.mean()/b.mean()-1
        saved = next(x for x in evidence['screen']['effects'] if x['metric'] == metric)
        np.testing.assert_allclose(ci, saved['bonferroni_97_5_percent_ci'], rtol=1e-12)
        np.testing.assert_allclose(gain, saved['relative_gain'], rtol=1e-12)
        assert gain >= .01 and ci[0] > 0
        print(f'{metric}: gain={gain:.10%}, paired 97.5% CI={ci}, positive seeds={sum(d>0)}/20')
    assert np.mean([r['service_fairness'] for r in cand]) >= .92
    assert np.mean([r['bounds']['stale_upper'] for r in cand]) <= np.mean([r['bounds']['stale_lower'] for r in ref])
    assert all(evidence['screen']['checks'].values())
    print('Compact statistical evidence verified. Raw frame physics and mobility require the archived raw evidence.')


if __name__ == '__main__':
    main()
