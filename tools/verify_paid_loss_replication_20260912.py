"""Independent arithmetic verification of saved trial and report artifacts."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/paid_loss_replication_20260912'
def sha(p):
    with (ROOT/p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
r=json.loads((OUT/'results.json').read_text())
c=json.loads((ROOT/'config/paid_loss_replication_20260912.json').read_text())
manifest=json.loads((OUT/'MANIFEST.json').read_text())
for p,h in manifest.items():assert sha(p)==h,p
assert len(r['rows'])==40
assert {x['seed'] for x in r['rows']}==set(c['development_seeds'])
assert not set(c['development_seeds'])&set(range(3900,3920))
for x in r['rows']:
    assert x['paid_control'] and x['data_error']==c['data_error']
    assert abs(sum(x['role_energy_j'].values())-x['energy_j'])<1e-8
    for record in x['records']:
        assert abs(sum(record['role_energy_j'].values())-record['energy_j'])<1e-12
        assert abs(sum(record['energy_by_node_j'])-record['energy_j'])<1e-12
        assert 0<=record['accepted_members']<=record['reports_success']<=record['reports_scheduled']
    records=x['records'];q=x['cohort'];b=x['bounds']
    assert [z['round'] for z in records]==list(range(1,x['rounds']+1))
    assert abs(sum(z['energy_j'] for z in records)-x['energy_j'])<1e-9
    assert abs(x['packets_per_j']-x['delivered']/x['energy_j'])<1e-9
    assert x['observed_birth_frames']+x['missing_birth_frames']==2000
    for k in ['generated','delivered','pending']:assert records[-1][k]==x[k]
    assert q['generated']==sum(q[k] for k in ['delivered','stale','death','overflow','pending'])
    if q['generated']:
        assert abs(b['delivery_upper']-b['delivery_lower']-q['pending']/q['generated'])<1e-12
        assert 0<=b['delivery_lower']<=b['delivery_upper']<=1
result=dict(status='saved_evidence_arithmetic_verified',trials=40,frames=sum(x['rounds'] for x in r['rows']),all_seeds_retained=True,full_requested_horizon_not_identified=True,training_started=False,verifier_sha256=sha('tools/verify_paid_loss_replication_20260912.py'))
with (OUT/'FINAL_VERIFICATION.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
