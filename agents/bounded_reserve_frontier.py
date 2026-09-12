"""Preserve minimum-energy allocation; defer at most one packet per cluster.

Factorial modes vary only reserve and packet-priority scoring on the same
two-point frontier. This is not a future-optimal planner.
"""
import numpy as np
from agents.feasible_service_dp import cluster_action as incumbent
from agents.temporal_reserve_dp import reserve_change


def fixed_total(base,cluster,table,total):
    result=np.zeros(base.n_nodes,dtype=np.int64)
    if total==0:return result
    ch=int(base.cluster_heads[cluster]);nodes=sorted(np.flatnonzero((base.cluster_of==cluster)&base.alive&(np.arange(base.n_nodes)!=ch)),key=lambda n:tuple(base.positions[n]))
    if not base.alive[ch] or table.head[cluster,total]>base.energy[ch]:return None
    if len(set(tuple(base.positions[n]) for n in nodes))!=len(nodes):return None
    caps=np.minimum(base.queue,base.cfg.n_max);dp=np.full(total+1,np.inf);dp[0]=0.;choices=[]
    for node in nodes:
        nxt=np.full(total+1,np.inf);pick=np.full(total+1,-1,int)
        for k in range(min(int(caps[node]),total)+1):
            cost=table.member[node,total,k]
            if cost>base.energy[node]:continue
            values=dp[:total+1-k]+cost;better=values<nxt[k:]
            nxt[k:][better]=values[better];pick[k:][better]=k
        dp=nxt;choices.append(pick)
    if not np.isfinite(dp[total]):return None
    remaining=total
    for node,pick in zip(reversed(nodes),reversed(choices)):
        k=int(pick[remaining]);assert k>=0;result[node]=k;remaining-=k
    assert remaining==0
    return result


def cluster_action(base,cluster,memory,table,mode):
    if mode not in ['reserve_only','priority_only','combined']:raise ValueError(mode)
    best=incumbent(base,cluster,table);total=int(best.sum())
    if total==0:return best
    alternative=fixed_total(base,cluster,table,total-1)
    if alternative is None:return best
    ch=int(base.cluster_heads[cluster]);nodes=np.flatnonzero((base.cluster_of==cluster)&base.alive&(np.arange(base.n_nodes)!=ch))
    replenishment=(base.cfg.packet_ttl_rounds+1)*memory.forecast()
    rates=np.divide(memory.delivered,memory.offered,out=np.zeros(base.n_nodes),where=memory.offered>0)
    deficit=np.maximum(0.,float(rates[nodes].mean())-rates) if len(nodes) else rates*0
    def score(action):
        quantity=int(action.sum());value=float(quantity)
        if mode in ['priority_only','combined']:
            for node in nodes:
                k=int(action[node]);value+=.5*sum(base.packet_ages[node][:k])/max(1,base.cfg.packet_ttl_rounds)+k*deficit[node]
        if mode in ['reserve_only','combined']:
            for node in nodes:
                value+=reserve_change(table.member[node,quantity,int(action[node])],base.energy[node],replenishment[node],1.)
            value+=reserve_change(table.head[cluster,quantity],base.energy[ch],replenishment[ch],1.)
        return value
    # Exact ties retain incumbent; no result-dependent threshold.
    return alternative if score(alternative)>score(best) else best
