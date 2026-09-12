"""Whole-cluster fixed-count energy DP with explicit QoS fallback."""
import numpy as np
from core.energy.idle_model import idle_listening_energy
from envs.step3_v3_env import episode_service_fairness


def allocation_energy(action,tx,idle):
    action=np.asarray(action); total=int(action.sum())
    return float(np.sum(action*tx+np.where(action>0,(total-action)*idle,0.)))


def minimum_energy_counts(tx,caps,total,idle):
    """Exact unconstrained fixed-count objective; caller supplies canonical order."""
    tx=np.asarray(tx,float); caps=np.asarray(caps,int)
    if not 0<=total<=caps.sum() or np.any(caps<0) or np.any(~np.isfinite(tx)):
        raise ValueError('invalid finite DP input')
    dp=np.full(total+1,np.inf); dp[0]=0.
    parent=np.full((len(caps),total+1),-1,dtype=np.int16)
    for i,cap in enumerate(caps):
        nxt=np.full_like(dp,np.inf)
        for k in range(min(int(cap),total)+1):
            cost=k*tx[i]+((total-k)*idle if k else 0.)
            values=dp[:total+1-k]+cost
            better=values<nxt[k:]
            nxt[k:][better]=values[better]; parent[i,k:][better]=k
        dp=nxt
    if not np.isfinite(dp[total]): raise RuntimeError('unreachable count')
    action=np.zeros(len(caps),dtype=np.int64); remaining=total
    for i in range(len(caps)-1,-1,-1):
        action[i]=parent[i,remaining]; remaining-=int(action[i])
    assert remaining==0 and action.sum()==total
    return action,float(dp[total])


def repair_action(env,base_action,mask,caps):
    base_action=np.asarray(base_action,dtype=np.int64); total=int(base_action.sum())
    audit=dict(changed=False,proposal_changed=False,reject_stale=False,reject_fairness=False,
               duplicate_state_fallback=False,predicted_saving_j=0.,proposal_saving_j=0.)
    nodes=np.flatnonzero(np.asarray(mask,bool)&(np.asarray(caps)>0))
    if total==0 or len(nodes)<2: return base_action.copy(),audit
    # Geometric canonical ordering avoids node-index tie preferences. Identical
    # observable keys fall back, rather than silently breaking equivariance.
    keys={int(n):tuple(map(float,env.base.positions[n])) for n in nodes}
    if len(set(keys.values()))!=len(nodes):
        audit['duplicate_state_fallback']=True; return base_action.copy(),audit
    nodes=np.array(sorted(nodes,key=lambda n:keys[int(n)]))
    tx=np.array([env.base.radio.tx(env.base.cfg.packet_bits,float(np.linalg.norm(env.base.positions[n]-env.base.positions[int(env.ch)]))) for n in nodes])
    idle=float(idle_listening_energy(1,p_idle_j_per_bit_time=env.base.cfg.e_elec_j_per_bit,slot_bit_times=env.base.cfg.idle_slot_bit_times))
    if not env.base.idle_energy_enabled: idle=0.
    local,opt=minimum_energy_counts(tx,np.asarray(caps)[nodes],total,idle)
    proposal=np.zeros_like(base_action); proposal[nodes]=local
    saving=allocation_energy(base_action[nodes],tx,idle)-opt
    assert saving>=-1e-12
    audit['proposal_changed']=bool(np.any(proposal!=base_action)); audit['proposal_saving_j']=max(0.,saving)
    def expiring(a):
        return sum(sum(age>=env.base.cfg.packet_ttl_rounds for age in env.base.packet_ages[n][:int(a[n])]) for n in nodes)
    # Same projected-demand definition as the existing B5/B6 guard.
    offered=env.step3_episode_offered_per_node+env.base.queue
    old_fair=episode_service_fairness(env.step3_episode_delivered_per_node+base_action,offered)
    new_fair=episode_service_fairness(env.step3_episode_delivered_per_node+proposal,offered)
    audit['reject_stale']=bool(expiring(proposal)<expiring(base_action))
    audit['reject_fairness']=bool(new_fair+1e-12<old_fair)
    audit.update(base_projected_fairness=old_fair,proposal_projected_fairness=new_fair)
    if audit['reject_stale'] or audit['reject_fairness'] or saving<=1e-12:
        return base_action.copy(),audit
    assert np.all(proposal<=caps) and np.all(proposal[~mask]==0) and proposal.sum()==total
    audit['changed']=True; audit['predicted_saving_j']=max(0.,saving)
    return proposal,audit
