"""Frozen CPU-parallel paired development experiment, exclusive output paths."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import gzip,hashlib,itertools,json,re,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from core.fragment_repair_fixed import make_tape,evaluate,POLICIES


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def job(seed,c,scenarios,out):
    tape=make_tape(seed,c['horizon']);sums={p:{} for p in POLICIES};filename=f'seed_{seed}.jsonl.gz'
    with gzip.open(Path(out)/filename,'xt',encoding='utf-8') as f:
        for i,scenario in enumerate(scenarios):
            for policy in POLICIES:
                r=evaluate(tape,policy,arrival_p=c['arrival_p'],**scenario);r.pop('records')
                f.write(json.dumps(dict(seed=seed,scenario=i,policy=policy,metrics=r))+'\n')
                for k,v in r.items():sums[policy][k]=sums[policy].get(k,0)+v
    means={p:{k:v/len(scenarios) for k,v in z.items()} for p,z in sums.items()}
    return dict(seed=seed,file=filename,mean=means)


def main():
    c=json.loads((ROOT/'config/fragment_repair_fixed_20260920.json').read_text())
    seeds=list(range(c['seed_start'],c['seed_start']+c['seed_count']))
    assert not set(seeds)&(set(range(3900,3920))|set(range(9700000,9700064)))
    registered_files=[];overlap=[]
    wanted=set(seeds)|{c['bootstrap_seed']}
    for directory in ('config','reports','tools','experiments'):
        for p in (ROOT/directory).glob('*'):
            if p.suffix not in ('.json','.py','.md') or 'fragment_repair' in p.name.lower():continue
            numbers={int(x) for x in re.findall(r'(?<!\d)\d{7}(?!\d)',p.read_text(encoding='utf-8',errors='replace'))}
            registered_files.append(str(p.relative_to(ROOT)))
            if numbers&wanted:overlap.append(dict(path=str(p),seeds=sorted(numbers&wanted)))
    assert not overlap,overlap
    for folder in ('fragment_repair_20260920','presence_screen_20260920','recovery_screen_20260920','chronological_screen_20260920'):
        for name in ('SOURCE_FREEZE.json','MANIFEST.json'):
            for p,h in json.loads((ROOT/'outputs'/folder/name).read_text()).items():assert sha(ROOT/p)==h,p
    release=ROOT/'releases/HTA_MAC_PAPER_BASELINE_V1_20260901'
    vf=json.loads((release/'ARTIFACT_MANIFEST.json').read_text())['files']
    for x in vf:assert sha(release/x['release_path'])==x['sha256']
    out=ROOT/'outputs/fragment_repair_fixed_20260920';out.mkdir(exist_ok=False)
    (out/'SEED_AUDIT.json').write_text(json.dumps(dict(seeds=seeds,bootstrap_seed=c['bootstrap_seed'],
        checked_registered_files=registered_files,overlap=overlap,prohibited_seeds_excluded=True,
        scope='registered code/config/reports; not a claim to have scanned every historical binary'),indent=2))
    names=['core/fragment_repair_fixed.py','core/chronological_calendar.py','core/timed_sleep_mac.py',
        'validation/test_fragment_repair_fixed.py','config/fragment_repair_fixed_20260920.json',
        'reports/FRAGMENT_REPAIR_FIXED_PROTOCOL_20260920.md','tools/run_fragment_repair_fixed_20260920.py',
        'tools/analyze_fragment_repair_fixed_20260920.py']
    (out/'SOURCE_FREEZE.json').write_text(json.dumps({p:sha(ROOT/p) for p in names},indent=2))
    fields=('capacity_uj','harvest_uj','head_change','ttl','loss','ack_loss')
    scenarios=[dict(zip(fields,values)) for values in itertools.product(*(c[k] for k in fields))]
    (out/'SCENARIOS.json').write_text(json.dumps(scenarios,indent=2))
    start=time.perf_counter();rows=[]
    with ProcessPoolExecutor(max_workers=c['workers']) as pool:
        jobs=[pool.submit(job,seed,c,scenarios,str(out)) for seed in seeds]
        for future in as_completed(jobs):
            rows.append(future.result())
            if len(rows)%8==0:print(json.dumps(dict(completed=len(rows),total=len(seeds))),flush=True)
    rows.sort(key=lambda z:z['seed'])
    (out/'SEED_SUMMARIES.json').write_text(json.dumps(rows,indent=2))
    meta=dict(scenarios=len(scenarios),seeds=seeds,independent_units=len(seeds),
        episodes=len(scenarios)*len(seeds)*len(POLICIES),workers=c['workers'],
        seconds=time.perf_counter()-start,v1_payloads_verified=len(vf),invariant_violations=0,
        neural_training=False,original_gate_a_certified=False,original_gate_b_certified=False)
    (out/'RUN.json').write_text(json.dumps(meta,indent=2))
    traces={p:evaluate(make_tape(seeds[0],c['horizon']),p,trace=True) for p in POLICIES}
    (out/'TRACES.json').write_text(json.dumps(traces,indent=2))
    (out/'MANIFEST.json').write_text(json.dumps({str(p.relative_to(ROOT)):sha(p) for p in out.iterdir() if p.is_file()},indent=2))
    print(json.dumps(meta,indent=2),flush=True)


if __name__=='__main__':main()
