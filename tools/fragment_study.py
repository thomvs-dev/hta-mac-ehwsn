"""Portable fragment study runner and raw-evidence verifier. No historical outputs required."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, gzip, hashlib, itertools, json, re, subprocess, sys, time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
STUDIES={'fixed':'fragment_repair_fixed_20260920','guard':'fragment_repair_guard_20260920',
         'shared':'shared_fragment_20260921'}


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,x):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f: json.dump(x,f,indent=2)


def engine(study):
    if study=='fixed': from core import fragment_repair_fixed as module
    elif study=='guard': from core import fragment_repair_guard as module
    else: from core import shared_fragment as module
    return module


def job(seed,c,scenarios,study,out):
    module=engine(study); tapes={}
    with gzip.open(Path(out)/f'seed_{seed}.jsonl.gz','xt',encoding='utf-8') as f:
        for i,sc in enumerate(scenarios):
            kwargs=dict(sc); sources=kwargs.pop('sources',None)
            if sources not in tapes:
                tapes[sources]=module.make_tape(seed,c['horizon'],sources) if sources else module.make_tape(seed,c['horizon'])
            for policy in c['policies']:
                z=module.evaluate(tapes[sources],policy,arrival_p=c['arrival_p'],**kwargs); z.pop('records')
                f.write(json.dumps(dict(seed=seed,scenario=i,policy=policy,metrics=z))+'\n')
    return seed


def analyze(out,c,scenario_count):
    rows={p:[] for p in c['policies']}; total=0
    for seed in range(c['seed_start'],c['seed_start']+c['seed_count']):
        groups={p:[] for p in rows}; seen=set(); generated={}
        with gzip.open(out/f'seed_{seed}.jsonl.gz','rt',encoding='utf-8') as f:
            for line in f:
                z=json.loads(line); p=z['policy']; sc=z['scenario']; m=z['metrics']
                assert z['seed']==seed and (sc,p) not in seen
                seen.add((sc,p)); total+=1
                assert m['generated']==sum(m[k] for k in ('delivered','pending_undelivered','stale','overflow','death'))
                assert m['energy_uj']==sum(m[k] for k in ('background_uj','control_uj','data_uj'))
                assert m['data_uj']==sum(m[k] for k in ('member_uj','forward_uj','feedback_uj'))
                assert generated.setdefault(sc,m['generated'])==m['generated']
                groups[p].append(m)
        assert seen==set(itertools.product(range(scenario_count),rows))
        for p,group in groups.items(): rows[p].append({k:sum(z[k] for z in group)/len(group) for k in group[0]})
    agg={p:{k:float(np.mean([z[k] for z in group])) for k in group[0]} for p,group in rows.items()}
    for z in agg.values():
        z['delivery_ratio']=z['delivered']/z['generated']; z['packets_per_j']=z['delivered']/(z['energy_uj']*1e-6)
    draws=np.random.default_rng(c['bootstrap_seed']).integers(0,c['seed_count'],size=(c['bootstrap_replicates'],c['seed_count']))
    boot={p:np.array([[z['delivered'],z['energy_uj']] for z in group])[draws].mean(axis=1) for p,group in rows.items()}
    iterations={}
    for candidate,bases in c['comparators'].items():
        comparisons={}
        for base in bases:
            a,b=agg[candidate],agg[base]; x,y=boot[candidate],boot[base]
            samples={'delivery':x[:,0]/y[:,0]-1,'efficiency':(x[:,0]/x[:,1])/(y[:,0]/y[:,1])-1}
            points={'delivery':a['delivered']/b['delivered']-1,'efficiency':a['packets_per_j']/b['packets_per_j']-1}
            comparisons[base]={}
            for metric,sample in samples.items():
                assert np.isfinite(sample).all()
                comparisons[base][metric]=dict(relative_gain=points[metric],paired_95_interval=np.quantile(sample,[.025,.975]).tolist(),family_adjusted_lower=float(np.quantile(sample,c['lower_quantile'])))
        passed=all(z['family_adjusted_lower']>=c['minimum_relative_gain'] for comp in comparisons.values() for z in comp.values())
        iterations[candidate]=dict(status='PASS_necessary_component_screen' if passed else 'STOP_candidate_gate_failed',comparisons=comparisons)
    return dict(aggregate=agg,iterations=iterations,raw_rows_recomputed=total,
                independent_seed_units=c['seed_count'],full_network_gate_pass=False)


def verify_v1():
    release=ROOT/'releases/HTA_MAC_PAPER_BASELINE_V1_20260901'
    payloads=json.loads((release/'ARTIFACT_MANIFEST.json').read_text())['files']
    for z in payloads: assert sha(release/z['release_path'])==z['sha256'],z['release_path']
    commit=subprocess.check_output(['git','rev-parse','paper-baseline-v1-20260901^{}'],cwd=ROOT,text=True).strip()
    assert commit=='11ef88c336a18833b8511b30dba857ae8b831086'
    return len(payloads)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','verify']); parser.add_argument('study',choices=STUDIES)
    parser.add_argument('directory',type=Path); args=parser.parse_args()
    cpath=ROOT/'config'/f'{STUDIES[args.study]}.json'; c=json.loads(cpath.read_text())
    out=args.directory.resolve(); v1=verify_v1()
    if args.action=='verify':
        for name,h in json.loads((out/'MANIFEST.json').read_text()).items(): assert sha(out/name)==h,name
        for name,h in json.loads((out/'SOURCE_FREEZE.json').read_text()).items(): assert sha(ROOT/name)==h,name
        scenarios=json.loads((out/'SCENARIOS.json').read_text()); result=analyze(out,c,len(scenarios))
        saved=json.loads((out/'results.json').read_text())
        for key in result: assert result[key]==saved[key],key
        print(json.dumps(dict(verified=True,v1_payloads=v1,episodes=result['raw_rows_recomputed'],iterations=result['iterations']),indent=2)); return
    seeds=list(range(c['seed_start'],c['seed_start']+c['seed_count']))
    assert not set(seeds)&set(range(3900,3920))
    audit=dict(seeds=seeds,bootstrap_seed=c['bootstrap_seed'],scope='registered text references; historical reruns explicitly reuse published seeds')
    if args.study=='shared':
        overlap=[]; checked=[]; wanted=set(seeds)|{c['bootstrap_seed']}
        for folder in ('config','reports','tools','experiments'):
            for p in (ROOT/folder).glob('*'):
                if p.suffix not in ('.py','.json','.md') or 'shared_fragment' in p.name.lower(): continue
                numbers={int(x) for x in re.findall(r'(?<!\d)\d{7}(?!\d)',p.read_text(encoding='utf-8',errors='replace'))}
                checked.append(str(p.relative_to(ROOT)))
                if numbers&wanted: overlap.append(str(p.relative_to(ROOT)))
        assert not overlap,overlap
        audit.update(checked=checked,overlap=overlap)
    out.mkdir(parents=True,exist_ok=False)
    paths=[cpath,Path(__file__),ROOT/'core/fragment_repair_fixed.py',ROOT/'core/chronological_calendar.py',ROOT/'core/timed_sleep_mac.py']
    if args.study=='shared': paths += [ROOT/'core/shared_fragment.py',ROOT/'validation/test_shared_fragment.py',ROOT/'reports/SHARED_FRAGMENT_PROTOCOL_20260921.md']
    if args.study=='guard': paths += [ROOT/'core/fragment_repair_guard.py']
    write(out/'SOURCE_FREEZE.json',{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in paths})
    write(out/'SEED_AUDIT.json',audit)
    fields=(['sources'] if args.study=='shared' else [])+['capacity_uj','harvest_uj','head_change','ttl','loss','ack_loss']
    scenarios=[dict(zip(fields,x)) for x in itertools.product(*(c[k] for k in fields))]
    write(out/'SCENARIOS.json',scenarios); start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=c['workers']) as pool:
        futures=[pool.submit(job,s,c,scenarios,args.study,str(out)) for s in seeds]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            if i%8==0: print(f'{i}/{len(seeds)} seeds complete',flush=True)
    run=dict(seeds=seeds,scenarios=len(scenarios),episodes=len(seeds)*len(scenarios)*len(c['policies']),workers=c['workers'],seconds=time.perf_counter()-start,v1_payloads_verified=v1,neural_training=False)
    write(out/'RUN.json',run); result=analyze(out,c,len(scenarios)); write(out/'results.json',result)
    write(out/'MANIFEST.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(json.dumps(dict(run=run,iterations=result['iterations'],metrics={p:{k:z[k] for k in ('delivery_ratio','packets_per_j')} for p,z in result['aggregate'].items()}),indent=2))


if __name__=='__main__': main()
