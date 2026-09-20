import numpy as np
import pytest
from core.shared_fragment import evaluate, make_tape, fragment, parse_fragment, parse_ack, ACK


def test_source_identity():
    packet=fragment(2,3,0)
    assert parse_fragment(packet,2,3)==0
    with pytest.raises(ValueError): parse_fragment(packet,1,3)
    with pytest.raises(ValueError): parse_ack(ACK.pack(3,2,4,63),1,3,4)


@pytest.mark.parametrize('policy',['whole','selective','block'])
def test_shared_ledger_and_fixed_windows(policy):
    tapes=make_tape(123,24,12)
    z=evaluate(tapes,policy,capacity_uj=64000,harvest_uj=1200,arrival_p=1,trace=True)
    assert z['generated']==288
    assert z['max_cache']<=2 and z['max_grants']<=24 and z['max_frame_us']<=1000000
    assert z['buffer_rejections']>0
    assert z['generated']==sum(z[k] for k in ['delivered','pending_undelivered','stale','death','overflow'])
    assert z['energy_uj']==z['control_uj']+z['data_uj']+z['background_uj']
    for frame in z['records']:
        events=frame['events']
        assert all(a['end_us']<=b['start_us'] for a,b in zip(events,events[1:]))
        for e in events:
            assert e['head_uj']==e['source_uj']+20
            assert e['source_uj']==int(np.ceil((e['end_us']-e['start_us'])*.05634))+2


def test_lost_receipts_retain_custody():
    tapes=make_tape(1,4,1)
    tapes[0]['arrivals'][:]=0
    z=evaluate(tapes,'block',capacity_uj=100000,loss='clean',ack_loss=1,ttl=6,trace=True)
    assert z['overflow']==2 and z['delivered']==2
    assert len(z['records'][-1]['queues'][0])==2


def test_absorbing_death_no_harvest_revival():
    z=evaluate(make_tape(1,4,4),'block',capacity_uj=60,harvest_uj=100000,arrival_p=1)
    assert z['delivered']==0 and z['death']==16 and z['energy_uj']==6*60


def test_exogenous_switch_and_per_source_ledger():
    z=evaluate(make_tape(1,6,4),'block',capacity_uj=100000,loss='clean',head_change=True,arrival_p=1,trace=True)
    assert [r['head'] for r in z['records']]==[4,4,4,5,5,5]
    assert z['delivered']==24


def test_controls_scale_at_shared_head():
    a=evaluate(make_tape(1,1,1),'block',capacity_uj=100000,arrival_p=0)
    b=evaluate(make_tape(1,1,4),'block',capacity_uj=100000,arrival_p=0)
    assert b['control_uj']>a['control_uj']*2
