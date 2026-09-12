import json
from pathlib import Path
import numpy as np
from experiments.evaluate_publication_ablation_and_all_cluster import build_environment
from envs.packet_cohort_audit import PacketCohort
from envs.reviewer_energy_feasibility_fast import admit


def test_passive_cohort_matches_uninstrumented_environment():
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/'config/feasible_service_frontier_20260906.json').read_text())
    a,_,_=build_environment(c,c['scenario'],902000,40)
    b,_,_=build_environment(c,c['scenario'],902000,40)
    ledger=PacketCohort(b,1,10)
    for step in range(40):
        action,_=admit(a,a.static_equal_action())
        a.step(action);b.step(action);ledger.observe(b,action)
        np.testing.assert_array_equal(a.energy,b.energy)
        np.testing.assert_array_equal(a.queue,b.queue)
        assert a.packet_ages==b.packet_ages
        assert a.total_packets_generated==b.total_packets_generated
    result=ledger.result();assert result['pending']==0
    assert result['generated']==sum(result[k] for k in ['delivered','stale','death','overflow'])


def test_cohort_records_overflow_stale_and_continued_arrivals():
    from types import SimpleNamespace
    base=SimpleNamespace(round=0,packet_ages=[[0]],np_random=np.random.default_rng(0),alive=np.array([True]),n_nodes=1,cfg=SimpleNamespace(packet_ttl_rounds=1,queue_max_packets=2))
    ledger=PacketCohort(base,0,1)
    ledger.recorder.arrivals=[3];base.round=1;base.packet_ages=[[0,0]]
    ledger.observe(base,np.array([0]))
    ledger.recorder.arrivals=[0];base.round=2;base.packet_ages=[[1,1]]
    ledger.observe(base,np.array([0]))
    ledger.recorder.arrivals=[2];base.round=3;base.packet_ages=[[0,0]]
    ledger.observe(base,np.array([0]))
    r=ledger.result();assert r['generated']==4 and r['overflow']==2 and r['stale']==2 and r['pending']==0
