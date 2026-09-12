"""Audit the frozen simulator's physical and endpoint definitions, no policy changes."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import concurrent.futures
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.run_v2_phase_a_fresh_audit_20260905 import (
    trainer,load_agent,build_transfer_environments,set_cpu_contract,validate_action,
    capped_energy_proportional_action,ROLE_FIELDS,sha256,seed_inventory,OnlinePrimalDualQoS,Step3QoSConstraintConfig)
OUT=ROOT/'outputs/reviewer_foundation_20260906'
CONTRACT=ROOT/'config/reviewer_foundation_20260906.json'


def deficit_audit(energy,consumed,harvested,delivered,cluster_of,heads):
    pre=np.asarray(consumed)>np.asarray(energy)+1e-12
    net=np.asarray(consumed)>np.asarray(energy)+np.asarray(harvested)+1e-12
    pre_associated=net_associated=0
    for node in np.flatnonzero(delivered):
        ch=int(heads[int(cluster_of[node])])
        pre_associated+=int(delivered[node])*int(pre[node] or pre[ch])
        net_associated+=int(delivered[node])*int(net[node] or net[ch])
    return dict(pre_harvest_deficit_nodes=int(pre.sum()),even_with_harvest_deficit_nodes=int(net.sum()),
                pre_harvest_associated_packets=pre_associated,even_with_harvest_associated_packets=net_associated,
                unbacked_consumption_j=float(np.maximum(0,consumed-energy-harvested).sum()))


def task(policy,seed):
    c=json.loads(CONTRACT.read_text());set_cpu_contract(1,seed)
    envs=build_transfer_environments(dict(c,development_seeds=[seed]),c['scenario'],trace=None)
    qos=Step3QoSConstraintConfig.from_payload(json.loads((ROOT/c['qos_config']).read_text()))
    agent=load_agent(ROOT/c['source_checkpoint'])[0] if policy=='hta_mac_v1' else None
    episodes=[]
    for env in envs:
        obs,mask,_=env.reset();dual=OnlinePrimalDualQoS(qos,c['primal_dual'],24);done=False
        totals={k:0 for k in ['pre_harvest_deficit_nodes','even_with_harvest_deficit_nodes','pre_harvest_associated_packets','even_with_harvest_associated_packets','unbacked_consumption_j']}
        records=[]; energy=0.; max_ledger=0
        while not done:
            state,active,caps=trainer.padded_state(env,obs,mask,env.base.n_nodes)
            if agent is not None: action,_=agent.act(state,active,epsilon=0.,caps=caps,budget=24)
            elif policy=='residual_energy': action=capped_energy_proportional_action(env,active,caps,24,c['energy_proportional_score_exponent'])
            else: action=dual.action(env,active,caps)
            validate_action(action,active,caps,24)
            clusters=env.base.cluster_of.copy(); heads=env.base.cluster_heads.copy()
            obs,mask,done,info=env.step(action);tr=info['energy_trace']
            a=deficit_audit(tr['energy_before'],tr['consumed'],tr['harvested'],info['delivered_packets_per_node'],clusters,heads)
            for k,v in a.items():totals[k]+=v
            if a['pre_harvest_deficit_nodes']:records.append(dict(round=int(env.base.round),**a))
            energy+=float(np.sum(tr['consumed']))
            ledger=env.base.total_packets_generated-(env.base.total_packets+env.base.dropped_stale_packets+env.base.dropped_death_packets+env.base.dropped_overflow_packets+int(env.base.queue.sum()))
            max_ledger=max(max_ledger,abs(int(ledger)));assert ledger==0
        q=env.step3_qos_counts
        episodes.append(dict(seed=seed,policy=policy,rank=int(env.target_rank),rounds=int(env.base.round),
             backlog_service_ratio=q['delivered']/max(1,q['demand']),
             unique_generated=env.base.total_packets_generated,global_delivered=env.base.total_packets,
             generated_packet_delivery_so_far=env.base.total_packets/max(1,env.base.total_packets_generated),
             pending_at_stop=int(env.base.queue.sum()),stale_packets=env.base.dropped_stale_packets,
             death_packets=env.base.dropped_death_packets,overflow_packets=env.base.dropped_overflow_packets,
             packets_per_j=env.base.total_packets/energy,packet_ledger_max_error=max_ledger,
             deficit_round_records=records,**totals))
    result=dict(policy=policy,seed=seed,episodes=episodes)
    with (OUT/f'{policy}_{seed}.json').open('x') as f:json.dump(result,f,indent=2)
    return result


def main():
    c=json.loads(CONTRACT.read_text()); inv=seed_inventory(c,CONTRACT);assert inv['fresh_development_cohort']
    for p,h in c['preserved_sha256'].items():assert sha256(p)==h
    assert not (OUT/'results.json').exists(); start=time.perf_counter();rows=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(task,p,s) for s in c['development_seeds'] for p in c['policies']]
        for job in concurrent.futures.as_completed(jobs):
            rows.append(job.result());print('DONE',len(rows),60,flush=True)
    for p,h in c['preserved_sha256'].items():assert sha256(p)==h,p
    affected=sum(e['even_with_harvest_associated_packets'] for r in rows for e in r['episodes'])
    result=dict(status='foundation_failure_requires_versioned_physics_repair' if affected else 'no_sampled_unbacked_packet_credit',
                affected_packet_credits=affected,rows=rows,seed_inventory=inv,elapsed_seconds=time.perf_counter()-start,
                old_gates_unchanged=True,training_started=False,contract_sha256=sha256(CONTRACT))
    with (OUT/'results.json').open('x') as f:json.dump(result,f,indent=2)
    print('STATUS',result['status'],affected,flush=True)


if __name__=='__main__':main()
