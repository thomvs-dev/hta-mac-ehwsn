"""Conditional full-reporting cost envelope under the inherited radio law.

All geometrical members participate, including nodes that may be dead in the
MAC trajectory. Thus this is a protocol demand envelope, not realized energy,
admission feasibility, or a hardware-specific radio simulation.
"""
import numpy as np


def full_reporting_cost(positions, heads, radio, bits):
    pos=np.asarray(positions,dtype=float);heads=np.asarray(heads,dtype=int)
    if pos.ndim!=2 or pos.shape[1]!=2 or not np.isfinite(pos).all():raise ValueError('invalid positions')
    if len(heads)==0 or len(set(heads))!=len(heads) or np.any(heads<0) or np.any(heads>=len(pos)):raise ValueError('invalid heads')
    if type(bits) is not int or bits<=0:raise ValueError('positive integer message size required')
    distances=np.linalg.norm(pos[:,None,:]-pos[heads][None,:,:],axis=2)
    cluster=distances.argmin(axis=1)
    for ci,ch in enumerate(heads):cluster[ch]=ci
    role={k:np.zeros(len(pos)) for k in ['report_tx','report_rx','grant_tx','grant_rx','control_listen']}
    max_ack=2*radio.rx(bits);links=0
    for ci,ch in enumerate(heads):
        members=np.flatnonzero((cluster==ci)&~np.isin(np.arange(len(pos)),heads))
        for node in members:
            d=float(distances[node,ci]);tx=radio.tx(bits,d);rx=radio.rx(bits)
            role['report_tx'][node]+=tx;role['report_rx'][ch]+=rx
            role['grant_tx'][ch]+=tx;role['grant_rx'][node]+=rx
            # Optional control-phase listening: all other report/grant messages.
            # It is additional to the inherited data-phase idle term.
            role['control_listen'][node]+=2*max(0,len(members)-1)*rx
            max_ack=max(max_ack,tx+rx);links+=1
    exchange=sum(role[k] for k in ['report_tx','report_rx','grant_tx','grant_rx'])
    return dict(role_j=role,exchange_j=exchange,listen_j=role['control_listen'],members=links,ack_min_j=2*radio.rx(bits),ack_max_j=max_ack)
