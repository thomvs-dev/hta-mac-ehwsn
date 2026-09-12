from types import SimpleNamespace
import numpy as np
from core.energy.radio_model import RadioModel
from envs.reviewer_energy_feasibility import admit,frame_cost


def fixture(energy):
    cfg=SimpleNamespace(packet_bits=10,e_elec_j_per_bit=.01,idle_slot_bit_times=10,
        bs_position_m=(3.,4.),frame_slot_budget=4)
    return SimpleNamespace(n_nodes=3,positions=np.array([[0.,0.],[1.,0.],[0.,1.]]),
        cluster_heads=np.array([2]),cluster_of=np.zeros(3,int),alive=np.ones(3,bool),
        queue=np.array([3,3,3]),energy=np.array(energy,float),cfg=cfg,idle_energy_enabled=True,
        radio=RadioModel(.01,.001,.0001,.002,10.))


def scalar_reference(base,action):
    # Independently expand the radio law and idle charges; never call frame_cost
    # or RadioModel methods. This is a model-consistency check, not hardware.
    result=[0.,0.,0.]; total=sum(action[:2])
    for node in [0,1]:
        count=action[node]
        if count:
            d=float(np.linalg.norm(base.positions[node]-base.positions[2]))
            amp=.001*d*d if d<10 else .0001*d**4
            result[node]=10*count*(.01+amp)+(total-count)*.1
    if total:
        d=float(np.linalg.norm(base.positions[2]-np.array([3.,4.])))
        result[2]=10*total*(.01+.002)+10*(.01+.001*d*d)
    return np.array(result)


def test_independent_radio_and_idle_reference():
    b=fixture([10,10,10])
    for i in range(4):
        for j in range(4-i+1):
            a=np.array([i,j,0]);np.testing.assert_allclose(frame_cost(b,a),scalar_reference(b,a),atol=1e-14)


def test_sender_receiver_affordability():
    for energy in [[0,1,1],[1,1,.01],[.2,.2,.8],[10,10,10]]:
        b=fixture(energy);a,record=admit(b,np.array([2,2,0]))
        assert np.all(scalar_reference(b,a)<=b.energy+1e-14)
        assert sum(a)<=4 and record['removed_slots']==4-sum(a)


def test_harvest_is_not_an_admission_input_and_dead_mask():
    b=fixture([0,0,0]);b.future_harvest=np.ones(3)*100
    a,_=admit(b,np.array([2,2,0]));assert not a.any()
    b=fixture([10,10,10]);b.alive[2]=False
    a,_=admit(b,np.array([2,2,0]));assert not a.any()


def test_full_transition_against_scalar_energy_balance():
    b=fixture([.2,.2,.8]);a,_=admit(b,np.array([2,2,0]))
    cost=scalar_reference(b,a); harvest=np.array([.03,.01,.02])
    expected=np.array([min(1.,b.energy[i]-cost[i]+harvest[i]) for i in range(3)])
    from core.energy.idle_model import energy_update
    np.testing.assert_allclose(energy_update(b.energy,frame_cost(b,a),harvest,1.),expected,atol=1e-14)
