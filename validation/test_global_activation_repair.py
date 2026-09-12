import itertools
import numpy as np
from agents.global_activation_repair import minimum_energy_counts,allocation_energy
from agents.global_activation_repair import repair_action
from types import SimpleNamespace


def test_dp_matches_exhaustive():
    rng=np.random.default_rng(999010)
    for _ in range(30):
        caps=rng.integers(1,4,4); tx=rng.uniform(.0001,.003,4); idle=.0002
        total=int(rng.integers(caps.sum()+1))
        action,value=minimum_energy_counts(tx,caps,total,idle)
        brute=min(allocation_energy(np.array(a),tx,idle) for a in itertools.product(*[range(c+1) for c in caps]) if sum(a)==total)
        assert abs(value-brute)<1e-12
        assert abs(value-allocation_energy(action,tx,idle))<1e-12


def test_empty_and_single():
    a,c=minimum_energy_counts([],[],0,.1); assert len(a)==0 and c==0
    a,c=minimum_energy_counts([.2],[3],2,.1); assert a.tolist()==[2] and abs(c-.4)<1e-12


def test_permutation_unique_optimum():
    tx=np.array([.11,.23,.37]); caps=np.array([3,3,3]); p=np.array([2,0,1])
    a,v=minimum_energy_counts(tx,caps,4,.05)
    b,w=minimum_energy_counts(tx[p],caps[p],4,.05)
    np.testing.assert_array_equal(b,a[p]); assert abs(v-w)<1e-12


def test_activation_cost_prefers_packing():
    a,value=minimum_energy_counts([.01,.01],[3,3],3,.1)
    assert np.count_nonzero(a)==1 and abs(value-.03)<1e-12


def test_fairness_fallback_and_safe_acceptance():
    cfg=SimpleNamespace(packet_bits=1,e_elec_j_per_bit=.1,idle_slot_bit_times=1,packet_ttl_rounds=3)
    base=SimpleNamespace(positions=np.array([[0.,0.],[1.,0.],[2.,0.]]),
                         radio=SimpleNamespace(tx=lambda bits,d:bits*(.01+d*.001)),cfg=cfg,
                         idle_energy_enabled=True,packet_ages=[[1,1,1],[1,1,1],[]],queue=np.array([3,3,0]))
    env=SimpleNamespace(base=base,ch=2,step3_episode_offered_per_node=np.array([10,10,0]),
                        step3_episode_delivered_per_node=np.array([0,0,0]))
    mask=np.array([1,1,0],bool); caps=np.array([3,3,0]); baseline=np.array([2,1,0])
    action,audit=repair_action(env,baseline,mask,caps)
    np.testing.assert_array_equal(action,baseline); assert audit['reject_fairness']
    env.step3_episode_delivered_per_node=np.array([10,0,0])
    action,audit=repair_action(env,baseline,mask,caps)
    assert audit['changed'] and action.tolist()==[0,3,0]
    env.base.packet_ages[0]=[3,3,3]
    action,audit=repair_action(env,baseline,mask,caps)
    assert audit['reject_stale']; np.testing.assert_array_equal(action,baseline)
