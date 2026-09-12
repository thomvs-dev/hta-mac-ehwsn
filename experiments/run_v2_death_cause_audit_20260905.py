"""Read-only policy replay with longitudinal death-role instrumentation."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import concurrent.futures
from collections import deque
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.run_v2_phase_a_fresh_audit_20260905 import (
    trainer,load_agent,build_transfer_environments,set_cpu_contract,validate_action,
    capped_energy_proportional_action,ROLE_FIELDS,sha256,seed_inventory)

OUT=ROOT/'outputs/phaseA/v2_death_cause_20260905'
CONTRACT=ROOT/'config/v2_death_cause_20260905.json'


def role_label(node,members,ch,heads):
    if node==ch: return 'target_ch'
    if node in members: return 'target_member'
    if node in heads: return 'background_ch'
    return 'background_member'


def death_records(before,after,members,ch,heads,energy,harvest,roles,controlled,window,round_number,first):
    records=[]
    for node in np.flatnonzero(before & ~after):
        local=role_label(int(node),members,ch,heads)
        total=roles[:,node]
        records.append(dict(node=int(node),round=round_number,role_at_death=local,
                            first_network_death=first,energy_before_j=float(energy[node]),harvest_on_death_round_j=float(harvest[node]),
                            cumulative_role_j=dict(zip(ROLE_FIELDS,map(float,total))),
                            cumulative_controlled_role_j=dict(zip(ROLE_FIELDS,map(float,controlled[:,node]))),
                            last_16_round_role_j=dict(zip(ROLE_FIELDS,map(float,window[:,node]))),
                            cumulative_controlled_fraction=float(controlled[:,node].sum()/max(total.sum(),1e-15))))
    return records


def task(policy,seed):
    c=json.loads(CONTRACT.read_text()); runtime=dict(c,development_seeds=[seed])
    set_cpu_contract(1,seed)
    envs=build_transfer_environments(runtime,c['scenario'],trace=None)
    agent=load_agent(ROOT/c['source_checkpoint'])[0] if policy=='hta_mac_v1' else None
    episodes=[]
    for env in envs:
        obs,mask,_=env.reset(); done=False; events=[]; history=deque(maxlen=16)
        roles=np.zeros((5,env.base.n_nodes)); controlled=roles.copy(); first_seen=False; err=0.
        while not done:
            state,active,caps=trainer.padded_state(env,obs,mask,env.base.n_nodes)
            if agent is not None: action,_=agent.act(state,active,epsilon=0.,caps=caps,budget=24)
            else: action=capped_energy_proportional_action(env,active,caps,24,c['energy_proportional_score_exponent'])
            validate_action(action,active,caps,24)
            alive=env.base.alive.copy(); energy=env.base.energy.copy()
            members=env.members.copy(); ch=int(env.ch); heads=env.base.cluster_heads.copy()
            obs,mask,done,info=env.step(action)
            step_roles=np.stack([info['energy_trace']['role_energy'][key] for key in ROLE_FIELDS])
            err=max(err,float(np.max(np.abs(step_roles.sum(0)-info['energy_trace']['consumed']))))
            assert err<1e-12 and np.all(step_roles>=0)
            roles+=step_roles; control=np.zeros(env.base.n_nodes,bool); control[members]=True; control[ch]=True
            controlled+=step_roles*control[None,:]; history.append(step_roles)
            deaths=death_records(alive,env.base.alive,members,ch,heads,energy,
                                 info['energy_trace']['harvested'],roles,controlled,sum(history),int(env.base.round),not first_seen)
            if deaths: first_seen=True
            events.extend(deaths)
        counts=env.step3_qos_counts; demand=max(1,int(counts['demand']))
        local_last=[e for e in events if e['round']==env.base.round and e['role_at_death'].startswith('target_')]
        reason='local_death' if local_last else ('horizon' if env.base.round>=c['horizon'] else 'other')
        episodes.append(dict(seed=seed,policy=policy,target_rank=int(env.target_rank),end_round=int(env.base.round),
                             terminal_reason=reason,death_events=events,role_energy_j=dict(zip(ROLE_FIELDS,map(float,roles.sum(1)))),
                             delivery_ratio=int(counts['delivered'])/demand,stale_ratio=int(counts['stale'])/demand,
                             fairness=float(counts['episode_service_fairness']),rmst=int(env.base.t_fnd or c['horizon']),
                             packets_per_j=float(env.base.total_packets/max(roles.sum(),1e-12)),
                             max_role_reconstruction_error_j=err))
    record=dict(policy=policy,seed=seed,episodes=episodes)
    with (OUT/f'{policy}_{seed}.json').open('x') as f: json.dump(record,f,indent=2)
    return record


def main():
    c=json.loads(CONTRACT.read_text()); inventory=seed_inventory(c,CONTRACT)
    assert inventory['fresh_development_cohort']
    for name,digest in c['source_hashes'].items(): assert sha256(name)==digest
    pre=json.loads((OUT/'preflight.json').read_text())
    assert not (OUT/'results.json').exists()
    started=time.perf_counter(); records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(task,p,s) for s in c['development_seeds'] for p in c['policies']]
        for job in concurrent.futures.as_completed(jobs):
            r=job.result(); records.append(r); print('DONE',len(records),40,r['policy'],r['seed'],flush=True)
    for name,digest in pre['preserved_sha256'].items(): assert sha256(name)==digest,name
    for name,digest in c['source_hashes'].items(): assert sha256(name)==digest,name
    result=dict(status='death_role_diagnostic_complete_no_gate_advancement',records=records,
                elapsed_seconds=time.perf_counter()-started,contract_sha256=sha256(CONTRACT),
                preserved_inventory_verified=True,seed_inventory=inventory,
                gate_a_threshold_unchanged=.8,neural_training=False,reserved_seeds_opened=False)
    with (OUT/'results.json').open('x') as f: json.dump(result,f,indent=2)


if __name__=='__main__': main()
