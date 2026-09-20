"""Chronological paid-wire finite study, separate from all frozen environments.

One source, two possible exogenous heads, finite horizon. Conservative RX-power
envelopes are actually debited; both hops forward every payload byte. Policies
receive decoded reports only. This is not a TSCH/RPL implementation.
"""
from dataclasses import dataclass
from functools import lru_cache
from math import ceil
import struct
from core.timed_sleep_mac import wire_fragments

REPORT=struct.Struct('>IHHIBIIII')  # epoch, src, head, free uJ, count, two(id,deadline)
GRANT=struct.Struct('>IHHIBIII')   # epoch, src, head, seq, count, next slot, two ids
NONE=2**32-1
HORIZON=6
RESCUE=(0,3)


def encode_report(epoch,head,free,queue):
    if len(queue)>2:raise ValueError('report queue cap')
    entries=list(queue)+[(NONE,NONE)]*(2-len(queue))
    return REPORT.pack(epoch,0,head,free,len(queue),*entries[0],*entries[1])


def decode_report(packet,epoch,head):
    if len(packet)!=REPORT.size:raise ValueError('bad report length')
    e,source,h,free,n,i0,d0,i1,d1=REPORT.unpack(packet)
    if (e,source,h)!=(epoch,0,head) or n>2:raise ValueError('bad report identity')
    q=((i0,d0),(i1,d1))[:n]
    if len({x[0] for x in q})!=n:raise ValueError('duplicate reported packet')
    return free,q


def encode_grant(epoch,head,sequence,ids,next_slot):
    if len(ids)>2:raise ValueError('grant cap')
    return GRANT.pack(epoch,0,head,sequence,len(ids),next_slot,*(list(ids)+[NONE]*(2-len(ids))))


def decode_grant(packet,epoch,head,slot,valid_ids):
    if len(packet)!=GRANT.size:raise ValueError('bad grant length')
    e,source,h,seq,n,nxt,a,b=GRANT.unpack(packet)
    if (e,source,h)!=(epoch,0,head) or n>2 or not slot<nxt<=HORIZON:
        raise ValueError('bad grant identity or time')
    ids=(a,b)[:n]
    if len(set(ids))!=n or not set(ids)<=set(valid_ids):raise ValueError('grant not in current report')
    return ids,nxt,seq


@lru_cache(None)
def layout(slot,extra_wake_uj):
    guard=20+ceil(80*(slot+1));cursor=1192;events=[]
    def packet(sender,receiver,size,phase,attempt=-1):
        nonlocal cursor
        start=cursor;cursor+=2*guard+32*size
        events.append((start,cursor,sender,receiver,size,phase,attempt))
    packet('sink','both',33,'sync')
    packet('source','head',REPORT.size+17,'report')
    packet('head','source',GRANT.size+17,'grant')
    control_end=cursor;completions=[];ends=[]
    for k in range(2):
        for n in wire_fragments(500):packet('source','head',n,'member_data',k)
        for n in wire_fragments(500):packet('head','sink',n,'forward_data',k)
        completions.append(cursor)
        packet('sink','head',33,'sink_receipt',k)
        packet('head','source',33,'source_receipt',k)
        ends.append(cursor)
    assert cursor<1_000_000
    # Deep-floor60 uJ is paid separately for every alive node/frame.
    control_uj=ceil(control_end*.05634)+extra_wake_uj
    data_us=ends[0]-control_end
    return dict(control_uj=control_uj,source_attempt_uj=ceil(data_us*.05634)+2,
                head_attempt_uj=ceil(data_us*.05634)+22,
                events=tuple(events),control_end_us=control_end,
                forward_end_us=tuple(completions),finish_us=cursor,guard_us=guard)


@dataclass(frozen=True)
class View:
    slot:int
    queue:tuple
    source_free_uj:int
    head_free_uj:int


def decide(policy,view,extra_wake_uj):
    """No future arrivals, channel outcomes, sink ledger or future head input."""
    t=view.slot;q=tuple(sorted(view.queue,key=lambda x:(x[1],x[0])))
    boundary=3 if t<3 else HORIZON
    if isinstance(policy,int):
        nxt=next((s for s in range(t+1,HORIZON) if s in RESCUE or policy&(1<<s)),HORIZON)
        count=len(q)
    elif policy=='separate_edf':
        count=len(q);nxt=min(t+2,boundary)
    elif policy=='deadline_batch':
        # Joint present admission and next rendezvous, a deterministic heuristic.
        urgent=bool(q and q[0][1]<=t+1)
        count=len(q) if len(q)==2 or urgent or t==HORIZON-1 else 0
        nxt=min(t+(1 if len(q)>count else 2),boundary)
    else:raise ValueError('unknown policy')
    # Current report is fresh, all earlier commitments already subtracted.
    # Future control is funded now; no expected harvest is spent.
    next_cost=0 if nxt==HORIZON or nxt in RESCUE else layout(nxt,extra_wake_uj)['control_uj']
    model=layout(t,extra_wake_uj)
    while count and (count*model['source_attempt_uj']+next_cost>view.source_free_uj or
                     count*model['head_attempt_uj']+next_cost>view.head_free_uj):count-=1
    if next_cost>min(view.source_free_uj,view.head_free_uj):nxt=boundary
    return tuple(x[0] for x in q[:count]),nxt


def static_policies():
    return [sum(1<<s for s in RESCUE)+sum((1<<s) for j,s in enumerate((1,2,4,5)) if m&(1<<j)) for m in range(16)]


def evaluate(arrivals,good_slots,policy,*,capacity_uj=50000,harvest_uj=0,
             head_change=False,extra_wake_uj=10,fault='clean',initial_packet=False,trace=False):
    if fault not in ('clean','lost_grant','lost_receipt','lost_report','lost_sync'):
        raise ValueError('unknown control fault')
    battery=[capacity_uj]*3;alive=[capacity_uj>0]*3;spent=[0]*3;harvested=[0]*3;spill=[0]*3
    mandatory=[set(RESCUE) for _ in range(3)]
    commitments=[set() for _ in range(3)]
    source_wakes=set(RESCUE);head_wakes=[set(),set(RESCUE),set(RESCUE)]
    q=[];delivered=set();terminal={};births={};duplicates=0;attempts=0;seq=0
    observed_head=None;observed_epoch=None;records=[];new_id=0

    def death(n):
        if battery[n]==0:alive[n]=False
        if n==0 and not alive[n]:
            for p,_ in q:
                if p not in delivered:terminal[p]='death'
            q.clear()

    def pay(n,cost):
        if cost<0 or cost>battery[n]:raise ValueError('unfunded debit')
        battery[n]-=cost;spent[n]+=cost;death(n)

    def free(n,t):
        future=set(s for s in mandatory[n]|commitments[n] if s>t)
        protected=sum(layout(s,extra_wake_uj)['control_uj'] for s in future)
        protected+=60*(HORIZON-t-1)  # future sleep floor is protected too
        return max(0,battery[n]-protected)

    def admit_control(n,t):
        cost=layout(t,extra_wake_uj)['control_uj']
        # Pending current control cannot spend future obligations.
        if alive[n] and free(n,t)>=cost:
            pay(n,cost);return alive[n]
        return False

    def add_packet(t):
        nonlocal new_id
        pid=new_id;new_id+=1;births[pid]=t
        if not alive[0]:terminal[pid]='death'
        elif len(q)>=2:terminal[pid]='overflow'
        else:q.append((pid,t+2))

    if initial_packet:add_packet(0)
    for t in range(HORIZON):
        # True exogenous schedule is used only by the environment/physical roles.
        head=2 if head_change and t>=3 else 1;epoch=int(head_change and t>=3)
        for n in range(3):
            if alive[n]:pay(n,min(60,battery[n]))
        for pid,deadline in list(q):
            if deadline<=t:
                q.remove((pid,deadline))
                if pid not in delivered:terminal[pid]='stale'
        if arrivals&(1<<t):add_packet(t)
        src_awake=t in source_wakes;ctrl_ok=[False]*3
        for n in range(3):
            scheduled=src_awake if n==0 else t in head_wakes[n]
            if scheduled:ctrl_ok[n]=admit_control(n,t)
            commitments[n].discard(t);mandatory[n].discard(t)
        m=layout(t,extra_wake_uj);ids=();nxt=None;report_bytes=None;grant_bytes=None
        physical_control=ctrl_ok[0] and ctrl_ok[head]
        sync_ok=physical_control and not(fault=='lost_sync' and t==1)
        if sync_ok:observed_head,observed_epoch=head,epoch
        report_ok=sync_ok and observed_head==head and observed_epoch==epoch and not(fault=='lost_report' and t==1)
        if report_ok:
            report_bytes=encode_report(epoch,head,free(0,t),q)
            source_free,reported=decode_report(report_bytes,epoch,head)
            view=View(t,reported,source_free,free(head,t))
            ids,nxt=decide(policy,view,extra_wake_uj)
            # Head reserves its next receiver obligation BEFORE sending grant.
            if nxt<HORIZON and nxt not in RESCUE:
                cost=layout(nxt,extra_wake_uj)['control_uj']
                assert cost+len(ids)*m['head_attempt_uj']<=free(head,t)
                commitments[head].add(nxt);head_wakes[head].add(nxt)
            assert len(ids)*m['head_attempt_uj']<=free(head,t)
            pay(head,len(ids)*m['head_attempt_uj'])
            seq+=1;grant_bytes=encode_grant(epoch,head,seq,ids,nxt)
            grant_ok=not(fault=='lost_grant' and t==1)
            if grant_ok:
                selected,next_slot,_=decode_grant(grant_bytes,observed_epoch,observed_head,t,[p for p,_ in q])
                future=0 if next_slot==HORIZON or next_slot in RESCUE else layout(next_slot,extra_wake_uj)['control_uj']
                assert len(selected)*m['source_attempt_uj']+future<=free(0,t)
                if future:commitments[0].add(next_slot)
                if next_slot<HORIZON:source_wakes.add(next_slot)
                # Budget is escrowed here; physical energy settles after events.
                charge=len(selected)*m['source_attempt_uj']
                for k,pid in enumerate(selected):
                    attempts+=1
                    deadline=next(d for p,d in q if p==pid)
                    on_time=t*1_000_000+m['forward_end_us'][k] < deadline*1_000_000
                    reached=bool(good_slots&(1<<t)) and on_time
                    if reached:
                        duplicates+=int(pid in delivered);delivered.add(pid)
                        # A lost source receipt cannot erase source-held payload.
                        if not(fault=='lost_receipt' and t==1):q[:]=[(p,d) for p,d in q if p!=pid]
                pay(0,charge)
        for n in range(3):
            if alive[n]:
                accepted=min(harvest_uj,max(0,capacity_uj-battery[n]))
                battery[n]+=accepted;harvested[n]+=accepted;spill[n]+=harvest_uj-accepted
            assert capacity_uj+harvested[n]==battery[n]+spent[n]
        assert len(q)<=2 and len(ids)<=2
        assert not(set(terminal)&delivered)
        assert set(births)==set(terminal)|delivered|{p for p,_ in q}
        assert alive[0] or not q
        if trace:records.append(dict(slot=t,actual_head=head,observed_head=observed_head,
            control_ok=ctrl_ok,report_hex=report_bytes.hex() if report_bytes else None,
            grant_hex=grant_bytes.hex() if grant_bytes else None,allocated=list(ids),next_slot=nxt,
            pending=list(q),delivered=sorted(delivered),battery_uj=list(battery),spent_uj=list(spent),
            events=m['events'][:3+12*len(ids)]))
    counts={label:sum(x==label for x in terminal.values()) for label in ('death','overflow','stale')}
    pending_undelivered=sum(p not in delivered for p,_ in q)
    assert sum(counts.values())+len(delivered)+pending_undelivered==len(births)
    return dict(delivered=len(delivered),generated=len(births),pending_undelivered=pending_undelivered,
        **counts,energy_uj=sum(spent),source_tx_attempts=attempts,duplicates=duplicates,
        final_battery_uj=battery,records=records)
