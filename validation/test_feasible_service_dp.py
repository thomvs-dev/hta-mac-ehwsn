import itertools
import numpy as np
from validation.test_reviewer_energy_feasibility import fixture,scalar_reference
from envs.reviewer_energy_feasibility import admit as reference,frame_cost
from envs.reviewer_energy_feasibility_fast import admit,CostTable
from agents.feasible_service_dp import cluster_action


def test_fast_admission_matches_reference_random_and_boundaries():
    rng=np.random.default_rng(901000)
    for i in range(150):
        b=fixture(rng.uniform(0,1.2,3));b.idle_energy_enabled=bool(i%2)
        if i%11==0:b.alive[2]=False
        if i%13==0:b.positions[1]=b.positions[0]
        a=np.array([rng.integers(0,5),0,0]);a[1]=rng.integers(0,5-a[0])
        x,rx=reference(b,a);y,ry=admit(b,a)
        np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(rx['predicted_cost_j'],ry['predicted_cost_j'])
        assert rx['removed_slots']==ry['removed_slots']


def test_dp_matches_exhaustive_scalar_feasible_frontier():
    rng=np.random.default_rng(901001)
    for _ in range(100):
        b=fixture(rng.uniform(0,1.2,3));a=cluster_action(b,0)
        feasible=[]
        for x,y in itertools.product(range(4),repeat=2):
            if x+y>4:continue
            v=np.array([x,y,0]);cost=scalar_reference(b,v)
            if np.all(cost<=b.energy):feasible.append((-(x+y),float(cost.sum())))
        optimum=min(feasible)
        assert -int(a.sum())==optimum[0]
        np.testing.assert_allclose(scalar_reference(b,a).sum(),optimum[1],rtol=0,atol=1e-14)


def test_multicluster_cost_and_admission_permutation():
    from types import SimpleNamespace
    rng=np.random.default_rng(901002);template=fixture([1,1,1])
    for _ in range(30):
        b=SimpleNamespace(**vars(template));b.n_nodes=6
        b.positions=rng.uniform(0,4,(6,2));b.cluster_heads=np.array([2,5]);b.cluster_of=np.array([0,0,0,1,1,1])
        b.energy=rng.uniform(0,1,6);b.queue=np.ones(6,dtype=int)*3;b.alive=np.ones(6,bool)
        a=np.array([2,2,0,2,2,0]);x,_=reference(b,a);y,_=admit(b,a)
        np.testing.assert_array_equal(x,y)
        np.testing.assert_array_equal(CostTable(b).costs(a)[0],frame_cost(b,a))
        perm=rng.permutation(6);inverse=np.argsort(perm)
        p=SimpleNamespace(**vars(b))
        for key in ['positions','energy','queue','alive','cluster_of']:setattr(p,key,getattr(b,key)[perm])
        p.cluster_heads=inverse[b.cluster_heads]
        z,_=admit(p,a[perm]);np.testing.assert_array_equal(z[inverse],y)
        np.testing.assert_array_equal(cluster_action(p,0)[inverse],cluster_action(b,0))
