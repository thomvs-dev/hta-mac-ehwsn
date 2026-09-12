import itertools
import numpy as np
from validation.test_reviewer_energy_feasibility import fixture,scalar_reference
from envs.reviewer_energy_feasibility_fast import CostTable
from agents.bounded_reserve_frontier import fixed_total,cluster_action
from agents.feasible_service_dp import cluster_action as incumbent
from agents.temporal_reserve_dp import ReserveMemory


def test_fixed_total_is_minimum_energy_by_enumeration():
    rng=np.random.default_rng(904000)
    for _ in range(60):
        b=fixture(rng.uniform(0,1.2,3));b.cfg.n_max=3;table=CostTable(b)
        for total in range(5):
            feasible=[scalar_reference(b,np.array([x,y,0])).sum() for x,y in itertools.product(range(4),repeat=2) if x+y==total and np.all(scalar_reference(b,np.array([x,y,0]))<=b.energy)]
            a=fixed_total(b,0,table,total)
            assert (a is not None)==bool(feasible)
            if feasible:np.testing.assert_allclose(scalar_reference(b,a).sum(),min(feasible),rtol=0,atol=1e-14)


def test_bounded_modes_preserve_energy_frontier_and_past_only_inputs():
    rng=np.random.default_rng(904001)
    for _ in range(60):
        b=fixture(rng.uniform(.01,1.2,3));b.cfg.n_max=3;b.cfg.packet_ttl_rounds=3;b.packet_ages=[[3,1,0],[2,1,0],[0]*3]
        table=CostTable(b);m=ReserveMemory(3);m.update(rng.uniform(0,.03,3),np.ones(3)*4,np.array([1,2,0]))
        original=incumbent(b,0,table)
        for mode in ['reserve_only','priority_only','combined']:
            a=cluster_action(b,0,m,table,mode)
            assert 0<=original.sum()-a.sum()<=1
            assert np.all(scalar_reference(b,a)<=b.energy+1e-14)
            expected=fixed_total(b,0,table,int(a.sum()))
            np.testing.assert_array_equal(a,expected)
            b.future_harvest=rng.uniform(0,100,3)
            np.testing.assert_array_equal(a,cluster_action(b,0,m,table,mode))
