"""Prospective all-seed evaluation over observed frozen HEART-CH replay."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import json
import hashlib
from pathlib import Path
import pickle
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/paid_loss_replication_20260912'
CONTRACT=ROOT/'config/paid_loss_replication_20260912.json'

def sha(path):
    with (ROOT/path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

from experiments.paid_control_trial import evaluate

def main():
    c=json.loads(CONTRACT.read_text())
    for p,h in c['preserved_sha256'].items():assert sha(p)==h,p
    from experiments.run_v2_phase_a_fresh_audit_20260905 import seed_inventory,set_cpu_contract
    inv=seed_inventory(c,CONTRACT);assert inv['fresh_development_cohort']
    OUT.mkdir(exist_ok=False);start=time.perf_counter();rows=[];schedules=[]
    try:
        from experiments.run_phase3_pilot import build_assets
        from core.ch_selection.frozen_schedule_full import frozen_ch_schedule_full
        set_cpu_contract(1,c['development_seeds'][0]);assets=build_assets(c['horizon'])
        import config as upstream
        assert Path(upstream.__file__).resolve()==(ROOT.parent/'final_repo/config.py').resolve()
        assert upstream.MOBILITY_ENABLED and upstream.MOBILE_FRACTION==.2
        for seed in c['development_seeds']:
            replay=frozen_ch_schedule_full(assets[0],seed,horizon=c['horizon'])
            positions=np.stack([f['positions'] for f in replay['frames']])
            assert np.isfinite(positions).all()
            moved=int(np.any(np.linalg.norm(np.diff(positions,axis=0),axis=2)>1e-12,axis=0).sum())
            path=OUT/f'heart_schedule_{seed}.pkl'
            with path.open('xb') as f:pickle.dump(replay,f,protocol=pickle.HIGHEST_PROTOCOL)
            schedules.append(dict(seed=seed,frames=len(replay['frames']),moving_nodes=moved,stop_reason=replay['stop_reason'],termination_cause=replay['upstream_termination_cause'],sha256=sha(path)))
            # A short replay is retained, not filtered. Empty cohorts remain
            # undefined and force an inconclusive/failed screening decision.
            for policy in c['policies']:rows.append(evaluate(c,seed,policy,replay,assets))
        for p,h in c['preserved_sha256'].items():assert sha(p)==h,p
        result=dict(status='complete_pending_censor_aware_screen',rows=rows,schedules=schedules,seed_inventory=inv,contract_sha256=sha(CONTRACT),elapsed_seconds=time.perf_counter()-start,training_started=False,historical_gates_unchanged=True)
        with (OUT/'results.json').open('x') as f:json.dump(result,f,indent=2)
    except Exception as e:
        with (OUT/'STOP_EVIDENCE.json').open('x') as f:json.dump(dict(status='integration_failure',error=repr(e),completed_rows=len(rows),schedules=schedules,training_started=False),f,indent=2)
        raise

if __name__=='__main__':main()
