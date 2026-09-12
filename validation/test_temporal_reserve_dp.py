import itertools
import numpy as np
from validation.test_reviewer_energy_feasibility import fixture,scalar_reference
from agents.temporal_reserve_dp import ReserveMemory,cluster_action


def test_temporal_dp_matches_exhaustive_scalar_objective():
    rng=np.random.default_rng(903000)
    for _ in range(50):
        b=fixture(rng.uniform(.05,1.2,3));b.cfg.n_max=3;b.cfg.packet_ttl_rounds=3
        b.packet_ages=[[3,1,0],[2,0,0],[0,0,0]]
        m=ReserveMemory(3);m.update(np.ones(3)*.01,np.ones(3),np.zeros(3))
        a=cluster_action(b,0,m)
        def value(v):
            cost=scalar_reference(b,v)
            if np.any(cost>b.energy+1e-14):return -np.inf
            reward=sum(sum(1.+.5*age/3 for age in b.packet_ages[i][:v[i]]) for i in [0,1])
            score=reward+sum(np.log(max(1e-12,(b.energy[i]+.04+1e-12-cost[i])/(b.energy[i]+.04+1e-12))) for i in range(3))
            return score
        best=max(value(np.array([x,y,0])) for x,y in itertools.product(range(4),repeat=2) if x+y<=4)
        np.testing.assert_allclose(value(a),best,rtol=0,atol=1e-12)


def test_history_causality_and_energy_constraint():
    b=fixture([.2,.2,.3]);b.cfg.n_max=3;b.cfg.packet_ttl_rounds=3;b.packet_ages=[[0]*3 for _ in range(3)]
    m=ReserveMemory(3);a=cluster_action(b,0,m)
    b.future_harvest=np.ones(3)*100;b.future_cluster_heads=[0,1]
    np.testing.assert_array_equal(a,cluster_action(b,0,m))
    assert np.all(scalar_reference(b,a)<=b.energy+1e-14)
    m.update(np.ones(3)*.1,np.ones(3),np.zeros(3))
    np.testing.assert_allclose(m.forecast(),.1)
    b.alive[2]=False;assert not cluster_action(b,0,m).any()


def test_temporal_permutation_and_duplicate_fallback():
    from types import SimpleNamespace
    b=fixture([.2,.5,.8]);b.cfg.n_max=3;b.cfg.packet_ttl_rounds=3;b.packet_ages=[[3,2,1],[2,1,0],[0]*3]
    m=ReserveMemory(3);m.update(np.array([.03,.01,.02]),np.ones(3)*3,np.array([1,2,0]))
    a=cluster_action(b,0,m);perm=np.array([1,2,0]);inverse=np.argsort(perm)
    p=SimpleNamespace(**vars(b))
    for key in ['positions','energy','queue','alive','cluster_of']:setattr(p,key,getattr(b,key)[perm])
    p.cluster_heads=inverse[b.cluster_heads];p.packet_ages=[b.packet_ages[i] for i in perm]
    pm=ReserveMemory(3)
    for key in ['mean','second','offered','delivered']:setattr(pm,key,getattr(m,key)[perm])
    np.testing.assert_array_equal(cluster_action(p,0,pm)[inverse],a)
    b.positions[1]=b.positions[0];assert not cluster_action(b,0,m).any()
