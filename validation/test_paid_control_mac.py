from pathlib import Path
import numpy as np
import pytest
from core.energy.radio_model import RadioModel
from core.hmm.solar_hmm import HMMParameters
from envs.intra_cluster_mac_env import MACEnvironmentConfig
from envs.paid_control_mac import PaidControlMAC,KeyedTraffic
from envs.packet_cohort_audit import PacketCohort


def fixture(error=0.,harvest=0.):
    def hmm(n):return HMMParameters(np.eye(n),np.full(n,harvest),np.zeros(n),np.ones(n)/n,Path('synthetic'),'unit fixture')
    cfg=MACEnvironmentConfig(1.,4000,5e-8,4,3,6,12,.01,.005,(2.,2.),packet_ttl_rounds=3)
    radio=RadioModel(5e-8,1e-11,1.3e-15,5e-9,87.)
    b=PaidControlMAC(cfg,radio,hmm(8),hmm(4),data_error=error)
    frame=dict(positions=np.array([[0.,0.],[1.,0.],[0.,1.]]),cluster_heads=np.array([2]),solar_states=np.zeros(3,int),thermal_states=np.zeros(3,int),stgcn_embedding=np.zeros((3,1)))
    b.reset(seed=909120,frozen_snapshot={'schedule':[frame.copy() for _ in range(12)]})
    return b


def test_control_independent_endpoint_cost_and_no_hidden_energy():
    b=fixture();b.energy[0]=0.
    old=b.energy.copy();view,table=b.begin_control()
    assert not view.alive[0] and view.queue[0]==0 and view.energy[0]==0
    assert not hasattr(view,'np_random') and not hasattr(view,'frozen_schedule')
    # CH listens for an unpowered member; failure is not free at the receiver.
    assert b.report_cost[2]>0 and b.report_cost[0]==0
    np.testing.assert_allclose(old-b.energy,b.control_cost,atol=1e-15)
    with pytest.raises(ValueError):b.step(np.array([1,0,0]))


def test_failed_data_is_paid_and_fifo_ledger_balances():
    b=fixture(error=1.);ledger=PacketCohort(b,0,12)
    for _ in range(12):
        before=b.energy.copy();view,table=b.begin_control()
        a=np.minimum(view.queue,np.array([1,1,0]));expected=b.control_cost+table.costs(a)[0]
        _,_,_,_,info=b.step(a);served=info['delivered_packets_per_node']
        assert served.sum()==0
        np.testing.assert_allclose(before-b.energy,expected,rtol=0,atol=1e-15)
        ledger.observe(b,served)
    assert b.total_packets==0 and b.dropped_stale_packets>0
    q=ledger.result();assert q['generated']==sum(q[k] for k in ['delivered','stale','death','overflow','pending'])


def test_common_exogenous_draws_do_not_shift_with_alive_set():
    a=fixture(harvest=1.);b=fixture(harvest=1.)
    b.alive[0]=False
    ha=a._sample_harvest();hb=b._sample_harvest()
    np.testing.assert_array_equal(ha[1:],hb[1:]);assert hb[0]==0
    t=KeyedTraffic();u=KeyedTraffic()
    t.begin(123,4,np.array([True,True,True]),2.);u.begin(123,4,np.array([False,True,True]),2.)
    first=[t.poisson(2.) for _ in range(3)];second=[u.poisson(2.) for _ in range(2)]
    assert first[1:]==second


def test_reliable_delivery_and_absorbing_death():
    b=fixture(harvest=1.);b.alive[0]=False;b.energy[0]=0
    ledger=PacketCohort(b,0,12)
    for _ in range(4):
        view,table=b.begin_control();a=np.minimum(view.queue,np.array([0,1,0]))
        _,_,_,_,info=b.step(a)
        np.testing.assert_array_equal(info['delivered_packets_per_node'],a)
        assert not b.alive[0] and b.energy[0]==0
        ledger.observe(b,a)


def test_complete_trial_harness_records_paid_energy(monkeypatch):
    import uuid
    import experiments.paid_control_trial as trial
    # Windows sandbox cannot traverse pytest's mode-0700 temporary directories.
    # Create an ordinary, uniquely named workspace directory without cleanup.
    tmp_path=Path(__file__).resolve().parents[1]/'tmp'/('paid_control_smoke_'+uuid.uuid4().hex)
    tmp_path.mkdir()
    b=fixture();monkeypatch.setattr(trial,'ROOT',tmp_path)
    c=dict(horizon=12,cohort_first_birth=1,cohort_last_birth=4,control_bits=100,data_error=0.,output='.')
    assets=(None,b.solar,b.thermal,b.radio,b.cfg,None)
    row=trial.evaluate(c,909120,'complete_frontier',{'frames':b.frozen_schedule},assets)
    assert row['rounds']==12 and row['invariants_pass']
    assert row['role_energy_j']['report']>0 and row['role_energy_j']['ack']>0
    assert sum(row['role_energy_j'].values())==pytest.approx(row['energy_j'])
    assert all(sum(r['energy_by_node_j'])==pytest.approx(r['energy_j']) for r in row['records'])
