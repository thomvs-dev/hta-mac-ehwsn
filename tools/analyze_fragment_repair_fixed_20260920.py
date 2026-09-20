"""Recompute all means from raw episodes, paired-seed bootstrap, report gates."""
from pathlib import Path
import csv,gzip,hashlib,json,re
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'outputs/fragment_repair_fixed_20260920'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    for name in ('SOURCE_FREEZE.json','MANIFEST.json'):
        for p,h in json.loads((O/name).read_text()).items():assert sha(R/p)==h,p
    c=json.loads((R/'config/fragment_repair_fixed_20260920.json').read_text())
    run=json.loads((O/'RUN.json').read_text());saved=json.loads((O/'SEED_SUMMARIES.json').read_text())
    assert len(saved)==c['seed_count']==64 and run['scenarios']==96
    by_policy={p:[] for p in c['policies']};raw_count=0
    for seed_row in saved:
        groups={p:[] for p in c['policies']};seen=set()
        with gzip.open(O/seed_row['file'],'rt') as f:
            for line in f:
                z=json.loads(line);m=z['metrics'];p=z['policy'];raw_count+=1
                assert z['seed']==seed_row['seed']
                assert (z['scenario'],p) not in seen;seen.add((z['scenario'],p))
                assert m['generated']==sum(m[k] for k in ('delivered','pending_undelivered','stale','overflow','death'))
                assert m['energy_uj']==m['background_uj']+m['control_uj']+m['data_uj']
                assert m['data_uj']==m['member_uj']+m['forward_uj']+m['feedback_uj']
                groups[p].append(m)
        assert seen=={(i,p) for i in range(96) for p in c['policies']}
        for p,rows in groups.items():
            mean={k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}
            for k,v in mean.items():assert abs(v-seed_row['mean'][p][k])<1e-8
            by_policy[p].append(mean)
    assert raw_count==run['episodes']==24576
    aggregate={p:{k:float(np.mean([r[k] for r in rows])) for k in rows[0]} for p,rows in by_policy.items()}
    for z in aggregate.values():
        z['delivery_ratio']=z['delivered']/z['generated'];z['packets_per_j']=z['delivered']/(1e-6*z['energy_uj'])
    rng=np.random.default_rng(c['bootstrap_seed'])
    draws=rng.integers(0,64,size=(c['bootstrap_replicates'],64))
    boot={p:np.array([[r['delivered'],r['energy_uj']] for r in rows])[draws].mean(axis=1) for p,rows in by_policy.items()}
    iterations={}
    for candidate,bases in c['comparators'].items():
        comparisons={}
        for base in bases:
            a,b=aggregate[candidate],aggregate[base]
            d=boot[candidate][:,0]/boot[base][:,0]-1
            e=(boot[candidate][:,0]/boot[candidate][:,1])/(boot[base][:,0]/boot[base][:,1])-1
            comparison={}
            for metric,point,samples in [('delivery',a['delivered']/b['delivered']-1,d),('efficiency',a['packets_per_j']/b['packets_per_j']-1,e)]:
                assert np.isfinite(samples).all()
                comparison[metric]=dict(relative_gain=point,
                    paired_95_interval=np.quantile(samples,[.025,.975]).tolist(),
                    family_adjusted_lower=float(np.quantile(samples,c['lower_quantile'])))
            comparisons[base]=comparison
        passed=all(z[m]['family_adjusted_lower']>=c['minimum_relative_gain'] for z in comparisons.values() for m in ('delivery','efficiency'))
        iterations[candidate]=dict(status='PASS_necessary_component_screen' if passed else 'STOP_candidate_gate_failed',comparisons=comparisons)
    result=dict(run=run,aggregate=aggregate,iterations=iterations,bootstrap_replicates=c['bootstrap_replicates'],
        bootstrap_seed=c['bootstrap_seed'],lower_quantile=c['lower_quantile'],family_claims=10,
        raw_rows_recomputed=raw_count,independent_seed_units=64,
        scope='new finite fragment simulator, not full-network or RFC reproduction',full_network_gate_pass=False)
    with (O/'results.json').open('x') as f:json.dump(result,f,indent=2)
    with (O/'comparison.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['candidate','baseline','metric','relative_gain','95_lower','95_upper','family_lower'])
        for p,it in iterations.items():
            for b,comp in it['comparisons'].items():
                for m,z in comp.items():w.writerow([p,b,m,z['relative_gain'],*z['paired_95_interval'],z['family_adjusted_lower']])
    with (O/'seed_means.json').open('x') as f:json.dump(by_policy,f)
    log=(R/'tmp/fragment_fixed_validation_20260920.log').read_text()
    match=re.search(r'(\d+) passed, (\d+) warnings',log);assert match and int(match[1])>=557
    table=[]
    for p,z in aggregate.items():table.append(f"| {p} | {100*z['delivery_ratio']:.3f}% | {z['packets_per_j']:.4f} | {z['energy_uj']:.2f} |")
    effects=[]
    for p,it in iterations.items():
        effects.append(f"### {p}: {it['status']}\n")
        for b,z in it['comparisons'].items():
            d=z['delivery'];e=z['efficiency']
            effects.append(f"- vs {b}: delivery {100*d['relative_gain']:+.3f}% (95% interval {100*d['paired_95_interval'][0]:+.3f} to {100*d['paired_95_interval'][1]:+.3f}%; adjusted lower {100*d['family_adjusted_lower']:+.3f}%); packets/J {100*e['relative_gain']:+.3f}% (95% interval {100*e['paired_95_interval'][0]:+.3f} to {100*e['paired_95_interval'][1]:+.3f}%; adjusted lower {100*e['family_adjusted_lower']:+.3f}%).")
    report=f'''# Fragment repair cycle: results

Completed two prespecified iterations of a new MAC-only fragment mechanism
study. The preceding variable-window run is INVALID for performance claims;
this correction charges both nodes through the fixed scheduled receipt boundary.
Research, exact protocol, cost assumptions and stop rules are in
[FRAGMENT_REPAIR_FIXED_PROTOCOL_20260920.md](FRAGMENT_REPAIR_FIXED_PROTOCOL_20260920.md).
Block ACK and selective fragment recovery are established mechanisms; this
experiment is not an RFC8931 implementation or a novelty claim.

## Results

| Policy | Timely delivery ratio | Packets / charged J | Mean charged energy uJ |
|---|---:|---:|---:|
{chr(10).join(table)}

{chr(10).join(effects)}

Gate: BOTH family-adjusted lower bounds must be >=+1% against EVERY required
comparator. Block compares to whole/selective; prefix additionally compares to
block. Neither favorable individual metrics nor scenario subsets rescue failure.
Only component screening is possible here; original80% Gate A, original Gate B,
full-network performance and journal readiness are not certified.

## What was implemented

Paid serialized data fragments, cumulative bitmaps, grants and source reports;
two-hop persistent reassembly; absorbing death, queue/buffer cap2; end-to-end
custody confirmation; deterministic energy/airtime admission. Iteration1 replaces
per-fragment ACK with one ACK per hop block. Iteration2 reduces an unaffordable
block while preserving cached forwarding before new intake. The selective-repeat
baseline ALSO admits affordable partial windows. No future channel/harvest input.

The head cache is released based on the head's received sink confirmation,
not knowledge of whether the source got its final receipt. Head changes invalidate
source knowledge; fragments are not transferred to a new head for free.

## Execution and validation

- {run['episodes']:,} episodes,96 scenario cells,64 paired development seed units,
 8 workers,{run['seconds']:.3f}s. Seeds9700100--9700163; bootstrap9789998.
- {int(match[1])} tests passed,{int(match[2])} warnings. All raw packet/energy ledgers
 and saved seed means recomputed; source/output hashes verified. Zero rollout
 invariant failures. V1's{run['v1_payloads_verified']} frozen payloads verified.
- Paired bootstrap20000 repetitions over SEEDS, retaining scenario and policy
 pairing; no treatment of24576 episodes as independent samples. Family10claims,
 lower percentile0.005; bootstrap coverage is approximate. Ordinary95% intervals
 are descriptive, not substitutes for the adjusted gate.
- Horizon24x1s; capacities4000/16000uJ, harvest0/300uJ/frame, TTL2/6frames,
 arrivalp0.35, six fragments carrying500bytes, at most2 granted packets/frame.
 Data laws clean/independent0.9/hop-frame-burst0.9; ACK loss0/0.1.
- Source config and tests were frozen before performance results. Seed audit
 covers registered code/config/report references and explicitly excludes3900--3919;
 it does not claim to scan all historical binary files.

## Interpretation and stop

These are model measurements using conservative radio envelopes. There is no
hardware validation and no full HEART-CH evaluation. The mains-powered sink is
outside the battery denominator. All comparators receive the same physical
accounting and paid reliable sync/report/grant; data and ACK loss are tested.
The new fragment/error model and24-frame horizon preclude direct numerical
comparison with the earlier calendar aggregates or the100-node V1 results.

Stop every failed candidate; retain its evidence. A necessary component pass
would still require independent protocol verification and a separately frozen
full-network integration/confirmation study. No neural training was run. Do not
retune these opened seeds or portray known ARQ mechanisms as a new architecture.

## Reproduction and files

Branch codex/v2-fragment-repair-20260920. Core: core/fragment_repair_fixed.py; config:
config/fragment_repair_fixed_20260920.json. Run tools/run_fragment_repair_fixed_20260920.py,
then full `python -B -m pytest validation -q -p no:cacheprovider`, then
tools/analyze_fragment_repair_fixed_20260920.py. Scripts refuse existing evidence paths.
Use a new frozen directory for any new study; do not overwrite this run.

Machine evidence: outputs/fragment_repair_fixed_20260920/results.json, comparison.csv,
seed_*.jsonl.gz, SEED_SUMMARIES.json, seed_means.json, SCENARIOS.json, SEED_AUDIT.json,
SOURCE_FREEZE.json, MANIFEST.json, TRACES.json. Logs:
tmp/fragment_repair_fixed_20260920.log and tmp/fragment_fixed_validation_20260920.log.
'''
    rp=R/'reports/FRAGMENT_REPAIR_FIXED_RESULTS_20260920.md'
    with rp.open('x') as f:f.write(report)
    files=[rp,O/'results.json',O/'comparison.csv',O/'seed_means.json',R/'tmp/fragment_fixed_validation_20260920.log',R/'tmp/fragment_repair_fixed_20260920.log']
    with (O/'ANALYSIS_MANIFEST.json').open('x') as f:json.dump({str(p.relative_to(R)):sha(p) for p in files},f,indent=2)
    print(json.dumps(dict(aggregate=aggregate,iterations=iterations),indent=2))


if __name__=='__main__':main()
