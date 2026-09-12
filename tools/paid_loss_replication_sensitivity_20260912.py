"""Prespecified descriptive sensitivity; never changes the primary gate."""
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/paid_loss_replication_20260912'
c=json.loads((ROOT/'config/paid_loss_replication_20260912.json').read_text())
r=json.loads((OUT/'results.json').read_text())
spec=c['analysis_sensitivity'];rng=np.random.default_rng(spec['rng_seed'])
groups={p:sorted([x for x in r['rows'] if x['policy']==p],key=lambda x:x['seed']) for p in c['policies']}
a=groups['complete_frontier'];b=groups['reserve_only']
assert [x['seed'] for x in a]==[x['seed'] for x in b]==c['development_seeds']
effects={}
for metric in ['delivery','packets_per_j']:
    if metric=='delivery' and not all(x['bounds']['defined'] for x in a+b):
        effects[metric]={'status':'undefined_cohort_retained'};continue
    d=np.array([x['bounds']['delivery_lower']-y['bounds']['delivery_upper'] if metric=='delivery' else x['packets_per_j']-y['packets_per_j'] for x,y in zip(a,b)])
    boot=d[rng.integers(0,len(d),size=(spec['draws'],len(d)))].mean(axis=1)
    effects[metric]=dict(seed_differences=d.tolist(),percentile_97_5_ci=np.quantile(boot,[.0125,.9875]).tolist(),positive_seed_count=int((d>0).sum()))
with (OUT/'BOOTSTRAP_SENSITIVITY.json').open('x') as f:json.dump(dict(specification=spec,effects=effects,changes_primary_gate=False),f,indent=2)
print(json.dumps(effects))
