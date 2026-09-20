import pytest
from core.fragment_repair_fixed import (fragment,parse_fragment,ack_packet,parse_ack,
    CHUNKS,FULL,N,POLICIES,make_tape,evaluate,choose_plan,plan_cost,OVERHEAD)


def controlled(h=6):
    t=make_tape(123,h)
    for v in t.values():v.fill(0)
    t['arrivals'].fill(1);t['arrivals'][0]=0
    return t


def test_wire_payload_and_identity():
    assert sum(CHUNKS)==500 and N==6
    for i in range(N):
        packet=fragment(4,i)
        assert len(packet)+OVERHEAD<=133 and parse_fragment(packet,4)==i
        with pytest.raises(ValueError):parse_fragment(packet,5)
    packet=ack_packet(4,1,FULL)
    assert parse_ack(packet,4,1)==FULL
    with pytest.raises(ValueError):parse_ack(packet,4,2)
    with pytest.raises(ValueError):ack_packet(4,1,FULL+1)


def test_clean_all_schemes_deliver_payload_once():
    for p in POLICIES:
        r=evaluate(controlled(),p,loss='clean',ack_loss=0,capacity_uj=50000,harvest_uj=0)
        assert r['delivered']==r['generated']==1
        assert r['data_fragments']==2*N


def test_fragment_repair_preserves_progress_after_individual_loss():
    t=controlled();t['data'][0,0,0,2]=1
    for p in ('selective','block','prefix'):
        r=evaluate(t,p,ack_loss=0,ttl=6,capacity_uj=50000,harvest_uj=0,trace=True)
        assert r['delivered']==1
        assert r['records'][1]['events'][0]['member']==[2]
    w=evaluate(t,'whole',ack_loss=0,ttl=6,capacity_uj=50000,harvest_uj=0,trace=True)
    assert w['records'][1]['events'][0]['member']==list(range(N))


def test_lost_member_ack_does_not_expose_receiver_bitmap():
    t=controlled();t['ack'][0,0,0,N]=0;t['receipt'][0,0]=0
    # Set probabilities to succeed by default, then lose only these ACKs.
    t['ack'].fill(1);t['receipt'].fill(1);t['ack'][0,0,0,N]=0;t['receipt'][0,0]=0
    r=evaluate(t,'block',ack_loss=.1,capacity_uj=50000,harvest_uj=0,ttl=6,trace=True)
    assert r['records'][0]['queue']==[0]
    assert r['records'][1]['events'][0]['member']==list(range(N))
    assert r['delivered']==1


def test_lost_final_receipt_keeps_custody_but_can_recover_without_data():
    t=controlled();t['ack'].fill(1);t['receipt'].fill(1);t['receipt'][0,0]=0
    r=evaluate(t,'block',ack_loss=.1,capacity_uj=50000,harvest_uj=0,trace=True)
    assert r['records'][0]['queue']==[0] and r['records'][0]['delivered']==[0]
    e=r['records'][1]['events'][0]
    assert not e['member'] and not e['forward'] and not r['records'][1]['queue']


def test_partial_window_is_funded_and_whole_block_declines():
    full=plan_cost('block',tuple(range(N)),tuple(range(N)))
    budget=min(full[:2])-100
    assert choose_plan('block',0,0,0,budget,budget) is None
    chosen=choose_plan('prefix',0,0,0,budget,budget)
    assert chosen and 0<len(chosen[0])<N
    a,b,_=plan_cost('prefix',*chosen);assert a<=budget and b<=budget


def test_absorbing_death_and_harvest_conservation():
    t=controlled();t['arrivals'].fill(0)
    for p in POLICIES:
        r=evaluate(t,p,capacity_uj=0,harvest_uj=10000)
        assert r['death']==6 and r['delivered']==0 and r['energy_uj']==0


def test_all_schemes_faults_budgets_and_exogenous_head_changes():
    for seed in range(4):
        for p in POLICIES:
            for battery in (1000,4000,16000):
                for loss in ('clean','independent','burst'):
                    r=evaluate(make_tape(seed,24),p,capacity_uj=battery,head_change=True,loss=loss)
                    assert r['generated']==sum(r[k] for k in ('delivered','pending_undelivered','death','overflow','stale'))
                    assert r['energy_uj']==r['background_uj']+r['control_uj']+r['data_uj']


def test_ambiguous_receipt_cannot_shorten_reserved_radio_window():
    t=controlled();t['data'][0,0,0,:]=1
    t['ack'].fill(1);t['receipt'].fill(1);t['receipt'][0,0]=0
    r=evaluate(t,'block',ack_loss=.1,ttl=6,capacity_uj=50000,harvest_uj=0,trace=True)
    e=r['records'][0]['events'][0]
    a,b,duration=plan_cost('block',tuple(range(N)),tuple(range(N)))
    assert e['forward']==[] and e['end_us']-e['start_us']==duration
    assert e['source_uj']==a and e['head_uj']==b
    assert r['records'][0]['queue']==[0]
