"""Versioned pre-harvest admission kernel; not installed in frozen V1.

Input is a global per-node action. Call before service accounting, including
background actions. Silently inserting this inside the old wrapper is invalid:
that wrapper accounts requested service before calling its base environment.
"""
import numpy as np


def frame_cost(base,action):
    a=np.asarray(action,dtype=np.int64); cost=np.zeros(base.n_nodes)
    for ci,ch in enumerate(base.cluster_heads):
        nodes=np.flatnonzero((base.cluster_of==ci)&base.alive&(np.arange(base.n_nodes)!=ch)&(a>0))
        total=int(a[nodes].sum())
        if not total:continue
        for node in nodes:
            cost[node]=base.radio.tx(base.cfg.packet_bits*int(a[node]),float(np.linalg.norm(base.positions[node]-base.positions[ch])))
            if base.idle_energy_enabled:
                cost[node]+=(total-int(a[node]))*base.cfg.e_elec_j_per_bit*base.cfg.idle_slot_bit_times
        cost[ch]=base.radio.rx(total*base.cfg.packet_bits)+base.radio.aggregate(total*base.cfg.packet_bits)
        cost[ch]+=base.radio.tx(base.cfg.packet_bits,float(np.linalg.norm(base.positions[ch]-np.asarray(base.cfg.bs_position_m))))
    return cost


def admit(base,proposed):
    """Conservative feasibility filter, not a globally optimal scheduler.

    Only current batteries enter admission. Remove one packet at a time to
    reduce summed energy deficit, with canonical geometric tie ordering.
    Duplicate coordinates use an all-idle fallback to avoid identity ties.
    """
    raw=np.asarray(proposed)
    if raw.shape!=(base.n_nodes,) or not np.issubdtype(raw.dtype,np.integer) or np.any(raw<0):
        raise ValueError('invalid action')
    action=np.minimum(raw,base.queue).astype(np.int64)
    action[~base.alive]=0;action[base.cluster_heads]=0
    for ci,ch in enumerate(base.cluster_heads):
        members=base.cluster_of==ci
        if not base.alive[ch]:action[members]=0
        if action[members].sum()>base.cfg.frame_slot_budget:raise ValueError('proposal exceeds cluster budget')
    before=action.copy(); removed=0
    while True:
        cost=frame_cost(base,action)
        if np.all(cost<=base.energy):break
        candidates=np.flatnonzero(action)
        keys=[tuple(base.positions[n]) for n in candidates]
        if len(set(keys))!=len(keys):
            removed+=int(action.sum());action[:]=0;cost=frame_cost(base,action);break
        best=None; score=np.inf
        for node in sorted(candidates,key=lambda n:tuple(base.positions[n])):
            candidate=action.copy();candidate[node]-=1
            deficit=float(np.maximum(frame_cost(base,candidate)-base.energy,0.).sum())
            if deficit<score:
                best=candidate;score=deficit
        if best is None:raise RuntimeError('cannot resolve energy deficit')
        action=best;removed+=1
        assert removed<=int(before.sum())
    assert np.all(cost<=base.energy) and np.all(action<=before)
    return action,dict(removed_slots=removed,predicted_cost_j=cost.tolist())
