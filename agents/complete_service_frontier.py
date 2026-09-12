"""Maximize the existing reserve surrogate over all feasible service totals.

Only minimum-energy allocations at each total are considered. This optimizes
that finite surrogate family, not all allocations or long-run delivery.
"""
import numpy as np
from agents.bounded_reserve_frontier import fixed_total, cluster_action as bounded
from agents.feasible_service_dp import cluster_action as incumbent
from agents.temporal_reserve_dp import reserve_change


def score(base, cluster, memory, table, action):
    total=int(action.sum());ch=int(base.cluster_heads[cluster])
    nodes=sorted(np.flatnonzero((base.cluster_of==cluster)&base.alive&(np.arange(base.n_nodes)!=ch)),key=lambda n:tuple(base.positions[n]))
    replenishment=(base.cfg.packet_ttl_rounds+1)*memory.forecast()
    value=float(total)
    for node in nodes:
        value+=reserve_change(table.member[node,total,int(action[node])],base.energy[node],replenishment[node],1.)
    value+=reserve_change(table.head[cluster,total],base.energy[ch],replenishment[ch],1.)
    return value


def cluster_action(base, cluster, memory, motion, table):
    # The compatibility argument `motion` is deliberately unused. No future
    # CH persistence or future geometry is assumed by this allocator.
    best=bounded(base,cluster,memory,table,'reserve_only')
    full=incumbent(base,cluster,table)
    best_score=score(base,cluster,memory,table,best)
    for total in range(int(full.sum()),-1,-1):
        candidate=full if total==int(full.sum()) else fixed_total(base,cluster,table,total)
        if candidate is None:continue
        value=score(base,cluster,memory,table,candidate)
        if value>best_score:
            best=candidate;best_score=value
    return best
