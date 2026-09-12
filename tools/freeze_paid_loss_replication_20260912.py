"""Preregister independent replication without changing model or thresholds."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    with (ROOT/p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def seeds(x,key=''):
    if isinstance(x,dict):return set().union(*(seeds(v,k) for k,v in x.items())) if x else set()
    if isinstance(x,list):return set().union(*(seeds(v,key) for v in x)) if x else set()
    return {x} if type(x) is int and 'seed' in key.lower() else set()
source='config/paid_loss_20260912.json';c=json.loads((ROOT/source).read_text())
for p,h in c['preserved_sha256'].items():assert sha(p)==h,p
pipeline_path='outputs/paid_loss_20260912/PIPELINE_COMPLETE.json'
pipeline=json.loads((ROOT/pipeline_path).read_text())
assert pipeline['status']=='paid_loss_screen_pass_no_training_authorization'
for p,h in pipeline['sha256'].items():assert sha(p)==h,p
cohort=set(range(380000,380020));known=set();records=[]
for pattern in c['seed_registry_globs']:
    for p in ROOT.glob(pattern):
        values=seeds(json.loads(p.read_text(encoding='utf-8')));known.update(values)
        if values:records.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),seeds=sorted(values)))
assert not cohort&known and not cohort&set(range(3900,3920))
for n in ['paid_loss_replication_seed_search_20260912.txt','paid_loss_replication_seed_search_errors_20260912.txt']:
    assert not (ROOT/'tmp'/n).read_text(encoding='utf-8-sig').strip()
assert '250 passed' in (ROOT/'tmp/paid_loss_replication_validation_20260912.log').read_text()
pre='reports/PAID_LOSS_REPLICATION_PREFLIGHT_20260912.json'
with (ROOT/pre).open('x') as f:json.dump(dict(status='fresh_before_execution',development_seeds=sorted(cohort),registry=records,model_and_gate_unchanged=True,workers=1),f,indent=2)
c.update(status='frozen_independent_paid_loss_replication',development_seeds=sorted(cohort),output='outputs/paid_loss_replication_20260912',source_contract=source,cohort_role='independent replication, no model selection or tuning')
extra=[source,pre,pipeline_path,'tools/freeze_paid_loss_replication_20260912.py','experiments/evaluate_paid_loss_replication_20260912.py','tools/report_paid_loss_replication_20260912.py','tools/verify_paid_loss_replication_20260912.py','tools/paid_loss_replication_sensitivity_20260912.py','tools/finalize_paid_loss_replication_20260912.py','reports/PAID_LOSS_REPLICATION_METHOD_20260912.md','tmp/paid_loss_replication_validation_20260912.log']
c['preserved_sha256'].update({p:sha(p) for p in extra})
target=ROOT/'config/paid_loss_replication_20260912.json'
with target.open('x') as f:json.dump(c,f,indent=2)
print(json.dumps(dict(status=c['status'],seeds=c['development_seeds'],contract_sha256=sha(target))))
