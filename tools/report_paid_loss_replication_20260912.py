"""All-seed conservative screen for the separate observed-replay estimand."""
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/paid_loss_replication_20260912'
def sha(p):
    with (ROOT/p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    c=json.loads((ROOT/'config/paid_loss_replication_20260912.json').read_text());r=json.loads((OUT/'results.json').read_text())
    assert r['contract_sha256']==sha('config/paid_loss_replication_20260912.json')
    assert len(r['rows'])==len(c['development_seeds'])*len(c['policies'])
    for p,h in c['preserved_sha256'].items():assert sha(p)==h,p
    data={}
    for p in c['policies']:
        data[p]=sorted([x for x in r['rows'] if x['policy']==p],key=lambda x:x['seed'])
        assert [x['seed'] for x in data[p]]==c['development_seeds']
    for x in r['rows']:
        assert x==json.loads((OUT/f"{x['policy']}_{x['seed']}.json").read_text())
        q=x['cohort'];assert q['generated']==sum(q[k] for k in ['delivered','stale','death','overflow','pending'])
        assert x['generated']==sum(x[k] for k in ['delivered','stale','death','overflow','pending'])
        assert x['rounds']==len(x['records'])
    for z in r['schedules']:assert sha(OUT/f"heart_schedule_{z['seed']}.pkl")==z['sha256']
    cand=data['complete_frontier'];ref=data['reserve_only']
    defined=all(x['bounds']['defined'] for x in r['rows'])
    paired=all(len({x['rounds'] for x in r['rows'] if x['seed']==seed})==1 for seed in c['development_seeds'])
    checks=dict(all_seed_cohorts_defined=defined,matched_observed_horizons=paired,feasibility=all(x['invariants_pass'] for x in r['rows']),mobility=all(x['moving_nodes']==20 for x in r['schedules']))
    effects=[]
    if defined:
        def arr(rows,key,bounds=False):return np.array([x['bounds'][key] if bounds else x[key] for x in rows])
        for name,a,b in [('delivery',arr(cand,'delivery_lower',True),arr(ref,'delivery_upper',True)),('packets_per_j',arr(cand,'packets_per_j'),arr(ref,'packets_per_j'))]:
            d=a-b;half=float(t.ppf(1-.05/(2*2),len(d)-1)*d.std(ddof=1)/np.sqrt(len(d)))
            gain=float(a.mean()/b.mean()-1) if b.mean()>0 else None
            lo=float(d.mean()-half);hi=float(d.mean()+half)
            effects.append(dict(metric=name,relative_gain=gain,mean_paired_difference=float(d.mean()),bonferroni_97_5_percent_ci=[lo,hi],interpretation='candidate worst-case minus reference best-case pending-packet outcome' if name=='delivery' else 'observed-replay whole-run efficiency'))
            checks[name+'_gain']=gain is not None and gain>=c['screen']['relative_gain_min']
            checks[name+'_interval']=lo>0
        checks['fairness']=float(arr(cand,'service_fairness').mean())>=c['screen']['fairness_min']
        checks['stale_worst_case']=float(arr(cand,'stale_upper',True).mean())<=float(arr(ref,'stale_lower',True).mean())
    passed=all(checks.values())
    result=dict(status='observed_replay_screen_pass_not_training_authorization' if passed else 'STOP_observed_replay_screen_failed_or_inconclusive',checks=checks,effects=effects,trials=len(r['rows']),frames=sum(x['rounds'] for x in r['rows']),elapsed_seconds=r['elapsed_seconds'],training_started=False,historical_gates_unchanged=True,full_3000_frame_delivery_identified=False)
    with (OUT/'screen.json').open('x') as f:json.dump(result,f,indent=2)
    if not passed:
        with (OUT/'STOP_SCREEN.json').open('x') as f:json.dump(result,f,indent=2)
    lines=['# Censor-aware mobile HEART-CH evaluation — 12 September 2026','',f"Outcome: **{result['status']}**.",'','All twenty fresh seeds 380000–380019 are retained. The fixed birth window is 501–2500, but only births actually generated before replay termination are observed. Delivery bounds apply to these generated packets: delivered/generated through (delivered+pending)/generated. They do not identify outcomes of unborn packets, or full 3,000-frame performance. No inverse-censor weighting or independent-censor assumption is made.','',
    '| Seed | Policy | Observed frames | Missing birth-window frames | Generated cohort | Pending cohort | Delivery lower–upper | Packets/J | Fairness |', '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for p in c['policies']:
        for x in data[p]:
            q=x['cohort'];b=x['bounds'];span=f"{b['delivery_lower']:.5f}–{b['delivery_upper']:.5f}" if b['defined'] else 'undefined'
            lines.append(f"| {x['seed']} | {p} | {x['rounds']} | {x['missing_birth_frames']} | {q['generated']} | {q['pending']} | {span} | {x['packets_per_j']:.3f} | {x['service_fairness']:.5f} |")
    lines+=['','The frozen primary candidate is complete_frontier versus reserve_only. Only the frozen complete-frontier candidate and reserve-only comparator are evaluated. The new screen requires >=1% gain in conservative delivery and observed packets/J, positive Bonferroni-adjusted 97.5% paired t intervals across the two primary comparisons, mean fairness >=0.92, no worst-case mean stale increase, defined cohorts, matched observed horizons and zero checked feasibility errors. Twenty fresh seeds independently replicate the paid-loss model; this is not full-horizon or external-model confirmation. Both methods pay control and ACK costs. Exogenous traffic/harvest draws use fixed-size node-indexed arrays.','',f'Checks: `{checks}`.','']
    for e in effects:lines.append(f"- {e['metric']}: relative gain {e['relative_gain']}; paired difference {e['mean_paired_difference']}; adjusted interval {e['bonferroni_97_5_percent_ci']}.")
    lines+=['','Replay coverage and its upstream termination cause:']
    for z in r['schedules']:lines.append(f"- Seed {z['seed']}: {z['frames']} frames, {z['moving_nodes']} moving nodes, {z['stop_reason']}, {z['termination_cause']}.")
    lines+=['',f"Executed {result['trials']} trials / {result['frames']} observed frames in {result['elapsed_seconds']:.2f} seconds, using one CPU process under memory pressure. Source and replay hashes are retained. Tests and verification are recorded in the output directory.",'','The MAC uses revised pre-funded all-cluster service, fixed traffic and current frozen HEART-CH geometry/heads; it does not train CHs or change routing. One identical replay is reused across policies within each seed. The candidate maximizes the unchanged reserve score over all minimum-energy feasible-total allocations, including idle. It does not predict future cluster heads or geometry. Surrogate-score dominance is not a long-run performance guarantee. This separate model charges scheduled reports, grants and ACKs. Member data attempts and aggregate forwarding each have erasure probability 0.10. Control/ACK channels remain reliable when funded. Forward loss affects the whole received cluster prefix; full reserved radio costs are charged even on failure. No hardware evidence is claimed. Read PAID_LOSS_REPLICATION_METHOD_20260912.md for model changes and limits.','',
    'Missing birth frames are a truncation of the planned observation window, not a counted packet loss. Pending generated packets remain explicit. If the MAC itself terminates earlier, that row remains and the matched-horizon check prevents promotion. Policy-dependent generated counts, if present, are shown rather than suppressed. Initial battery transients, informative upstream stopping and heterogeneous observation lengths prevent extrapolation to sustained operation.','',
    'The earlier full-coverage gate remains failed; this evaluation has a different prospectively declared estimand and does not repair that gate. On failure stop without retuning, shortening the cohort, dropping seeds, changing thresholds, selecting a favorable ablation or starting training. V1 and historical Gate A/B remain unchanged.','',
    'Reproduction into absent paths: `python -B experiments/evaluate_paid_loss_replication_20260912.py`, then `python -B tools/report_paid_loss_replication_20260912.py`.']
    report=ROOT/'reports/PAID_LOSS_REPLICATION_RESULTS_20260912.md'
    with report.open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    paths=list(OUT.glob('*'))+[report,ROOT/'config/paid_loss_replication_20260912.json']
    with (OUT/'MANIFEST.json').open('x') as f:json.dump({str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()},f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
