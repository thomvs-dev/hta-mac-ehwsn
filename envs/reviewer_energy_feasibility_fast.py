"""Lookup/vector implementation of the preserved greedy admission reference."""
import numpy as np


class CostTable:
    def __init__(self,base):
        self.base=base; b=int(base.cfg.frame_slot_budget);self.b=b
        self.member=np.zeros((base.n_nodes,b+1,b+1))
        self.head=np.zeros((len(base.cluster_heads),b+1))
        for ci,ch in enumerate(base.cluster_heads):
            distance=float(np.linalg.norm(base.positions[ch]-np.asarray(base.cfg.bs_position_m)))
            for total in range(1,b+1):
                cost=base.radio.rx(total*base.cfg.packet_bits)+base.radio.aggregate(total*base.cfg.packet_bits)
                cost+=base.radio.tx(base.cfg.packet_bits,distance)
                self.head[ci,total]=cost
            for node in np.flatnonzero((base.cluster_of==ci)&base.alive&(np.arange(base.n_nodes)!=ch)):
                d=float(np.linalg.norm(base.positions[node]-base.positions[ch]))
                for k in range(1,b+1):
                    tx=base.radio.tx(base.cfg.packet_bits*k,d)
                    for total in range(k,b+1):
                        cost=tx
                        if base.idle_energy_enabled:
                            cost+=(total-k)*base.cfg.e_elec_j_per_bit*base.cfg.idle_slot_bit_times
                        self.member[node,total,k]=cost

    def costs(self,actions):
        a=np.atleast_2d(actions);base=self.base
        totals=np.stack([a[:,base.cluster_of==ci].sum(axis=1) for ci in range(len(base.cluster_heads))],axis=1)
        cost=self.member[np.arange(base.n_nodes)[None,:],totals[:,base.cluster_of],a].copy()
        cost[:,base.cluster_heads]=self.head[np.arange(len(base.cluster_heads))[None,:],totals]
        return cost


def admit(base,proposed):
    raw=np.asarray(proposed)
    if raw.shape!=(base.n_nodes,) or not np.issubdtype(raw.dtype,np.integer) or np.any(raw<0):raise ValueError('invalid action')
    action=np.minimum(raw,base.queue).astype(np.int64)
    action[~base.alive]=0;action[base.cluster_heads]=0
    for ci,ch in enumerate(base.cluster_heads):
        members=base.cluster_of==ci
        if not base.alive[ch]:action[members]=0
        if action[members].sum()>base.cfg.frame_slot_budget:raise ValueError('proposal exceeds cluster budget')
    before=int(action.sum());table=CostTable(base)
    while True:
        cost=table.costs(action)[0]
        if np.all(cost<=base.energy):break
        nodes=sorted(np.flatnonzero(action),key=lambda n:tuple(base.positions[n]))
        if len(set(tuple(base.positions[n]) for n in nodes))!=len(nodes):
            action[:]=0;cost=table.costs(action)[0];break
        candidates=np.repeat(action[None,:],len(nodes),axis=0)
        candidates[np.arange(len(nodes)),nodes]-=1
        deficit=np.maximum(table.costs(candidates)-base.energy,0.).sum(axis=1)
        action=candidates[int(np.argmin(deficit))]
    return action,dict(removed_slots=before-int(action.sum()),predicted_cost_j=cost.tolist())
