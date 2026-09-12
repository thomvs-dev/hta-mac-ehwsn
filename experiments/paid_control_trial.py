"""One paired-policy trial of the prospectively specified paid-control model."""
import json
from pathlib import Path
import time
import numpy as np
from envs.paid_control_mac import PaidControlMAC
from envs.packet_cohort_audit import PacketCohort
from envs.censored_packet_bounds import outcome_bounds
from envs.step3_v3_env import episode_service_fairness
from agents.temporal_reserve_dp import ReserveMemory
from agents.complete_service_frontier import cluster_action as frontier
from agents.bounded_reserve_frontier import cluster_action as reserve

ROOT=Path(__file__).resolve().parents[1]

def evaluate(c,seed,policy,replay,assets):
    _,solar,thermal,radio,cfg,_=assets
    b=PaidControlMAC(cfg,radio,solar,thermal,idle_energy_enabled=True,arrival_rate=1.,harvest_multiplier=1.,control_bits=c['control_bits'],data_error=c['data_error'])
    frames=replay['frames'];b.reset(seed=seed,frozen_snapshot={'schedule':frames})
    ledger=PacketCohort(b,c['cohort_first_birth'],c['cohort_last_birth']);memory=ReserveMemory(b.n_nodes)
    offered=np.zeros(b.n_nodes,dtype=np.int64);delivered=offered.copy();energy=0.;records=[];overrides=0;start=time.perf_counter()
    totals={k:0. for k in ['report','grant','member_tx','ch_rx','ch_aggregate','ch_forward','data_idle','ack']}
    for index in range(min(c['horizon'],len(frames))):
        np.testing.assert_array_equal(b.positions,frames[index]['positions'])
        np.testing.assert_array_equal(b.cluster_heads,frames[index]['cluster_heads'])
        before=b.energy.copy();alive=b.alive.copy();queue=b.queue.copy()
        eligible=np.zeros(b.n_nodes,bool)
        for ci,ch in enumerate(b.cluster_heads):
            if b.alive[ch]:eligible|=(b.cluster_of==ci)&b.alive&(np.arange(b.n_nodes)!=ch)
        view,table=b.begin_control()
        if index>0:
            mean=memory.mean.copy();second=memory.second.copy()
            # Only currently accepted reports reveal the previous harvest.
            memory.update(np.where(b.accepted,b.last_harvest,0.),np.zeros(b.n_nodes),np.zeros(b.n_nodes))
            memory.mean[~b.accepted]=mean[~b.accepted];memory.second[~b.accepted]=second[~b.accepted]
        a=np.zeros(b.n_nodes,dtype=np.int64)
        for ci in range(len(b.cluster_heads)):
            ref=reserve(view,ci,memory,table,'reserve_only')
            current=ref if policy=='reserve_only' else frontier(view,ci,memory,None,table)
            overrides+=int(not np.array_equal(current,ref));a+=current
        np.testing.assert_array_less(table.costs(a)[0]-b.energy,np.full(b.n_nodes,1e-14))
        _,_,terminated,truncated,info=b.step(a);served=info['delivered_packets_per_node'];trace=info['energy_trace']
        assert np.all(served<=a) and not np.any(b.alive&~alive)
        np.testing.assert_allclose(b.energy,np.minimum(cfg.initial_energy_j,np.maximum(0.,before-trace['consumed']+trace['harvested'])),rtol=0,atol=1e-12)
        ledger.observe(b,served);offered+=queue*eligible;delivered+=served
        assert b.total_packets_generated==b.total_packets+b.dropped_stale_packets+b.dropped_death_packets+b.dropped_overflow_packets+int(b.queue.sum())
        role={'report':trace['report'],'grant':trace['grant'],**trace['role_energy']}
        np.testing.assert_allclose(sum(role.values()),trace['consumed'],rtol=0,atol=1e-12)
        for k,v in role.items():totals[k]+=float(v.sum())
        energy+=float(trace['consumed'].sum())
        records.append(dict(round=int(b.round),generated=int(b.total_packets_generated),delivered=int(b.total_packets),pending=int(b.queue.sum()),energy_j=float(trace['consumed'].sum()),energy_by_node_j=trace['consumed'].tolist(),role_energy_j={k:float(v.sum()) for k,v in role.items()},attempted_packets=int(a.sum()),reports_success=info['report_success'],reports_scheduled=info['report_attempts'],accepted_members=info['accepted_members']))
        if terminated or truncated:break
    cohort=ledger.result();observed=max(0,min(b.round,c['cohort_last_birth'])-c['cohort_first_birth']+1)
    result=dict(seed=seed,policy=policy,rounds=int(b.round),schedule_coverage=len(frames),cohort=cohort,bounds=outcome_bounds(cohort),observed_birth_frames=observed,missing_birth_frames=c['cohort_last_birth']-c['cohort_first_birth']+1-observed,packets_per_j=b.total_packets/max(energy,1e-12),energy_j=energy,service_fairness=episode_service_fairness(delivered,offered),overrides=overrides,generated=int(b.total_packets_generated),delivered=int(b.total_packets),pending=int(b.queue.sum()),stale=int(b.dropped_stale_packets),death=int(b.dropped_death_packets),overflow=int(b.dropped_overflow_packets),invariants_pass=True,elapsed_seconds=time.perf_counter()-start,records=records,role_energy_j=totals,data_error=c['data_error'],paid_control=True)
    with (ROOT/c['output']/f'{policy}_{seed}.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ['seed','policy','rounds','bounds','packets_per_j']}),flush=True)
    return result
