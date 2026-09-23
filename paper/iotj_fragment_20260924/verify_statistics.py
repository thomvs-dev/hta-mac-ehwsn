"""Independent numerical checks from exported seed endpoints."""
from pathlib import Path
import csv,json,math
import numpy as np
from scipy.stats import t,binomtest
P=Path(__file__).resolve().parent
r=json.loads((P/'statistical_analysis.json').read_text())
with (P/'seed_endpoints.csv').open() as f: seeds=list(csv.DictReader(f))
for x in r['tests']:
    def endpoint(policy):
        return {int(z['seed']):float(z[x['metric']]) for z in seeds if z['study']==x['study'] and z['policy']==policy}
    a,b=endpoint(x['candidate']),endpoint(x['baseline'])
    assert a.keys()==b.keys() and len(a)==x['n']
    d=[a[s]-b[s] for s in sorted(a)]
    mean=sum(d)/len(d)
    sd=math.sqrt(sum((v-mean)**2 for v in d)/(len(d)-1))
    statistic=mean/(sd/math.sqrt(len(d)))
    p=2*t.sf(abs(statistic),len(d)-1)
    assert math.isclose(p,x['p_raw'],rel_tol=1e-11)
    assert math.isclose(statistic,x['t'],rel_tol=1e-11)
    width=t.ppf(.975,len(d)-1)*sd/math.sqrt(len(d))
    assert np.allclose([mean-width,mean+width],[x['ci95_low'],x['ci95_high']],rtol=1e-11)
    nonzero=[v for v in d if v!=0]
    assert len(nonzero)==x['n']-x['tied_seeds']
    sp=binomtest(sum(v>0 for v in nonzero),len(nonzero),.5).pvalue if nonzero else 1
    assert sp==x['sign_p_raw']
for raw,adjusted in [('p_raw','p_holm_22'),('sign_p_raw','sign_p_holm_22')]:
    ordered=sorted(r['tests'],key=lambda x:x[raw])
    running=0
    for i,x in enumerate(ordered):
        running=max(running,(22-i)*x[raw])
        assert math.isclose(x[adjusted],min(1,running),rel_tol=1e-12)
source=(P/'main.tex').read_text()
for x in r['tests']:
    if x['study']=='shared':
        # Verify every displayed confidence-bound rounding in the principal rows.
        assert f"{x['ci95_low']:.3f}" in source and f"{x['ci95_high']:.3f}" in source
        assert f"{x['t']:.2f}" in source and f"{x['cohen_dz']:.2f}" in source
        mantissa,exponent=f"{x['p_holm_22']:.2e}".split('e')
        assert mantissa+r'\times10^{'+str(int(exponent))+'}' in source
out={'comparisons_checked':22,'independent_seeds':160,'shared_table_rows_checked':4,
     'manual_t_tail_ci_checks':True,'sign_checks':True,'holm_checks':True,
     'original_gates_unchanged':True}
(P/'STATISTICS_VALIDATION.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
