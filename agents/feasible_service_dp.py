"""One-frame maximum feasible service, then minimum joules; no future inputs.

For each possible total service, active-member idle cost is fixed by that
total. Multiple-choice DP finds the least-energy feasible allocation. This
optimizes one frame only, not future service, fairness or long-run energy.
"""
import numpy as np
from envs.reviewer_energy_feasibility_fast import CostTable


def cluster_action(base,cluster,table=None):
    table=CostTable(base) if table is None else table
    result=np.zeros(base.n_nodes,dtype=np.int64);ch=base.cluster_heads[cluster]
    if not base.alive[ch]:return result
    nodes=sorted(np.flatnonzero((base.cluster_of==cluster)&base.alive&(np.arange(base.n_nodes)!=ch)),key=lambda n:tuple(base.positions[n]))
    # A conservative identity-independent fallback in geometrically ambiguous sets.
    if len(set(tuple(base.positions[n]) for n in nodes))!=len(nodes):return result
    caps=np.minimum(base.queue,getattr(base.cfg,'n_max',table.b))
    for total in range(min(table.b,int(caps[nodes].sum())),0,-1):
        if table.head[cluster,total]>base.energy[ch]:continue
        dp=np.full(total+1,np.inf);dp[0]=0.;choices=[]
        for node in nodes:
            nxt=np.full(total+1,np.inf);pick=np.full(total+1,-1,int)
            for k in range(min(int(caps[node]),total)+1):
                cost=table.member[node,total,k]
                if cost>base.energy[node]:continue
                candidate=dp[:total+1-k]+cost
                better=candidate<nxt[k:]
                nxt[k:][better]=candidate[better];pick[k:][better]=k
            choices.append(pick);dp=nxt
        if np.isfinite(dp[total]):
            remaining=total
            for node,pick in zip(reversed(nodes),reversed(choices)):
                k=int(pick[remaining]);assert k>=0;result[node]=k;remaining-=k
            assert remaining==0
            return result
    return result
