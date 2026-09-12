import itertools
import numpy as np
from agents.complete_service_frontier import cluster_action,score
from agents.bounded_reserve_frontier import cluster_action as bounded
from agents.temporal_reserve_dp import ReserveMemory
from envs.reviewer_energy_feasibility_fast import CostTable
from validation.test_reviewer_energy_feasibility import fixture,scalar_reference


def test_enumerated_minimum_energy_family_and_surrogate_dominance():
    rng=np.random.default_rng(906100)
    for _ in range(80):
        b=fixture(rng.uniform(.01,1.2,3));b.cfg.n_max=3;b.cfg.packet_ttl_rounds=3
        m=ReserveMemory(3);m.update(rng.uniform(0,.03,3),np.ones(3),np.zeros(3))
        table=CostTable(b);a=cluster_action(b,0,m,None,table)
        feasible=[]
        for i,j in itertools.product(range(4),repeat=2):
            x=np.array([i,j,0])
            if i+j<=4 and np.all(scalar_reference(b,x)<=b.energy):feasible.append(x)
        # Independent enumeration selects least-joule allocations at each T.
        family=[]
        for total in range(5):
            group=[x for x in feasible if x.sum()==total]
            if group:
                cost=min(scalar_reference(b,x).sum() for x in group)
                family.extend(x for x in group if abs(scalar_reference(b,x).sum()-cost)<1e-14)
        assert np.all(scalar_reference(b,a)<=b.energy+1e-14)
        assert score(b,0,m,table,a)>=max(score(b,0,m,table,x) for x in family)-1e-12
        reference=bounded(b,0,m,table,'reserve_only')
        assert score(b,0,m,table,a)>=score(b,0,m,table,reference)-1e-12
        b.future_harvest=np.full(3,1e9);b.future_heads=[0,1]
        np.testing.assert_array_equal(a,cluster_action(b,0,m,object(),table))
