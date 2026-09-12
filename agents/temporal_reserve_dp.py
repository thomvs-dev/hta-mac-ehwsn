"""Causal reserve-value architecture over the existing feasible MAC action set.

This is an analytical future-value surrogate, not a learned or optimal value
function. It combines service, deadline urgency, service deficit and a log
value for retained sender/CH energy. No future schedule or RNG is consulted.
"""
import numpy as np
from envs.reviewer_energy_feasibility_fast import CostTable


class ReserveMemory:
    def __init__(self,nodes):
        self.mean=np.zeros(nodes);self.second=np.zeros(nodes);self.steps=0
        self.offered=np.zeros(nodes);self.delivered=np.zeros(nodes)

    def update(self,harvest,offered,delivered):
        h=np.asarray(harvest);alpha=1. if self.steps==0 else .1
        self.mean=(1-alpha)*self.mean+alpha*h
        self.second=(1-alpha)*self.second+alpha*h*h
        self.offered+=offered;self.delivered+=delivered;self.steps+=1

    def forecast(self):
        # Conservative heuristic, not a calibrated confidence bound.
        return np.maximum(0.,self.mean-np.sqrt(np.maximum(0.,self.second-self.mean*self.mean)))


def reserve_change(cost,energy,replenishment,weight):
    scale=energy+replenishment+1e-12
    return weight*np.log(max(1e-12,(scale-cost)/scale))


def cluster_action(base,cluster,memory,table=None,reserve_weight=1.):
    table=CostTable(base) if table is None else table
    result=np.zeros(base.n_nodes,dtype=np.int64);ch=int(base.cluster_heads[cluster])
    if not base.alive[ch]:return result
    nodes=sorted(np.flatnonzero((base.cluster_of==cluster)&base.alive&(np.arange(base.n_nodes)!=ch)),key=lambda n:tuple(base.positions[n]))
    if not nodes or len(set(tuple(base.positions[n]) for n in nodes))!=len(nodes):return result
    caps=np.minimum(base.queue,base.cfg.n_max)
    replenishment=(base.cfg.packet_ttl_rounds+1)*memory.forecast()
    rates=np.divide(memory.delivered,memory.offered,out=np.zeros(base.n_nodes),where=memory.offered>0)
    target=float(np.mean(rates[nodes]));deficit=np.maximum(0.,target-rates)
    utility={}
    for node in nodes:
        ages=np.asarray(base.packet_ages[node][:int(caps[node])],float)
        utility[node]=np.r_[0.,np.cumsum(1.+.5*ages/max(1,base.cfg.packet_ttl_rounds)+deficit[node])]
    best_score=0. # All-idle is explicitly considered.
    for total in range(1,min(table.b,int(caps[nodes].sum()))+1):
        hc=table.head[cluster,total]
        if hc>base.energy[ch]:continue
        dp=np.full(total+1,-np.inf);dp[0]=0.;choices=[]
        for node in nodes:
            nxt=np.full(total+1,-np.inf);pick=np.full(total+1,-1,int)
            for k in range(min(int(caps[node]),total)+1):
                cost=table.member[node,total,k]
                if cost>base.energy[node]:continue
                value=utility[node][k]+reserve_change(cost,base.energy[node],replenishment[node],reserve_weight)
                candidate=dp[:total+1-k]+value;better=candidate>nxt[k:]
                nxt[k:][better]=candidate[better];pick[k:][better]=k
            dp=nxt;choices.append(pick)
        score=dp[total]+reserve_change(hc,base.energy[ch],replenishment[ch],reserve_weight)
        if score>best_score:
            best_score=float(score);result[:]=0;remaining=total
            for node,pick in zip(reversed(nodes),reversed(choices)):
                k=int(pick[remaining]);assert k>=0;result[node]=k;remaining-=k
            assert remaining==0
    return result
