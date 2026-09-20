"""Fixed-window two-hop MAC fragment-repair study, not an RFC implementation.

All schemes share paid reports, per-packet grants, persistent receiver buffers,
exogenous heads and conservative active-radio envelopes. No future observations.
"""
from functools import lru_cache
from math import ceil
import struct
import numpy as np
from core.chronological_calendar import encode_report,decode_report

HEADER=struct.Struct('>IHBB')  # datagram (birth frame), source, fragment, count
ACK=struct.Struct('>IHHB')    # datagram, source, head, cumulative bitmap
PLAN=struct.Struct('>IHHIBB') # epoch, source, head, datagram, member/forward masks
CHUNKS=tuple(min(99,500-i) for i in range(0,500,99))
N=len(CHUNKS);FULL=(1<<N)-1
POLICIES=('whole','selective','block','prefix')
FRAME_US=1_000_000;GUARD_US=400;TURN_US=192;OVERHEAD=26


def fragment(pid,index):
    if not 0<=index<N:raise ValueError('fragment index')
    return HEADER.pack(pid,0,index,N)+bytes([index])*CHUNKS[index]


def parse_fragment(packet,pid):
    if len(packet)<HEADER.size:raise ValueError('short fragment')
    p,s,i,n=HEADER.unpack(packet[:HEADER.size])
    if p!=pid or s!=0 or n!=N or not 0<=i<N or len(packet)!=HEADER.size+CHUNKS[i]:
        raise ValueError('invalid fragment')
    return i


def ack_packet(pid,head,mask):
    if mask&~FULL or mask<0:raise ValueError('invalid bitmap')
    return ACK.pack(pid,0,head,mask)


def parse_ack(packet,pid,head):
    if len(packet)!=ACK.size:raise ValueError('ACK length')
    p,s,h,mask=ACK.unpack(packet)
    if (p,s,h)!=(pid,0,head) or mask&~FULL:raise ValueError('ACK identity or bitmap')
    return mask


def bits(mask):return tuple(i for i in range(N) if mask&(1<<i))
def mask(indices):return sum(1<<i for i in indices)
def wire_time(size):return 2*GUARD_US+32*size+TURN_US
DATA_US=tuple(wire_time(HEADER.size+n+OVERHEAD) for n in CHUNKS)
ACK_US=wire_time(ACK.size+OVERHEAD)
PLAN_US=wire_time(PLAN.size+OVERHEAD)
CONTROL_US=1192+wire_time(33)+wire_time(29+2+OVERHEAD)
CONTROL_UJ=ceil(CONTROL_US*.05634)+10


def make_tape(seed,horizon):
    rng=np.random.default_rng(seed)
    return dict(arrivals=rng.random(horizon),data=rng.random((horizon,horizon,2,N)),
        ack=rng.random((horizon,horizon,2,N+1)),receipt=rng.random((horizon,horizon)),
        burst=rng.random((horizon,2)))


@lru_cache(None)
def plan_cost(policy,member,forward):
    count=len(member)+len(forward)
    ack_count=count if policy=='selective' else int(bool(member))+int(bool(forward))
    duration=PLAN_US+sum(DATA_US[i] for i in member+forward)+(ack_count+1)*ACK_US
    return ceil(duration*.05634)+2,ceil(duration*.05634)+22,duration


def choose_plan(policy,known_head,cached,known_sink,source_free,head_free):
    if policy not in POLICIES:raise ValueError('policy')
    member=bits(FULL if policy=='whole' else FULL&~known_head)
    forward=bits(FULL if policy=='whole' else (cached|mask(member))&~known_sink)
    if not member and not forward:
        return ((),()) if min(source_free-2,head_free-22)>=ceil((PLAN_US+ACK_US)*.05634) else None
    while True:
        if not member and not forward and known_sink!=FULL:return None
        cs,ch,_=plan_cost(policy,member,forward)
        if cs<=source_free and ch<=head_free:return member,forward
        if policy in ('whole','block'):return None
        # Both strong per-fragment ARQ and new block-prefix policy can reduce
        # the window. Preserve already cached forwarding before new intake.
        if member:
            member=member[:-1]
            forward=bits((cached|mask(member))&~known_sink)
        elif forward:forward=forward[:-1]
        else:return None


def evaluate(tape,policy,*,capacity_uj=16000,harvest_uj=300,head_change=False,
             ttl=2,loss='independent',ack_loss=.1,arrival_p=.35,trace=False):
    if policy not in POLICIES or loss not in ('clean','independent','burst'):raise ValueError('mode')
    horizon=len(tape['arrivals']);battery=[capacity_uj]*3;spent=[0]*3;harvested=[0]*3
    alive=[capacity_uj>0]*3;q=[];births=set();delivered=set();terminal={}
    caches=[{}, {}, {}];head_known_sink=[{}, {}, {}];source_known={};sink={}
    observed_head=None;counts=dict(data_fragments=0,ack_frames=0,duplicate_fragments=0,
        grant_frames=0,control_uj=0,data_uj=0,background_uj=0,member_uj=0,forward_uj=0,
        feedback_uj=0,partial_windows=0)
    records=[]

    def death(n):
        if battery[n]==0:alive[n]=False
        if not alive[n] and n>0:caches[n].clear();head_known_sink[n].clear()
        if not alive[0]:
            for p in q:
                if p not in delivered:terminal[p]='death'
            q.clear()

    def pay(n,cost,category):
        assert isinstance(cost,int) and 0<=cost<=battery[n]
        battery[n]-=cost;spent[n]+=cost;counts[category]+=cost;death(n)

    def available(n,t):return max(0,battery[n]-60*(horizon-t-1)) if alive[n] else 0
    def received(t,p,hop,i):
        return loss=='clean' or (tape['burst'][t,hop]<.9 if loss=='burst' else tape['data'][t,p,hop,i]<.9)

    for t in range(horizon):
        head=2 if head_change and t>=horizon//2 else 1
        for n in range(3):
            if alive[n]:pay(n,min(60,battery[n]),'background_uj')
        for p in list(q):
            if p+ttl<=t:
                q.remove(p)
                if p not in delivered:terminal[p]='stale'
        # At most two active reassemblies at each finite head. Expired/stale
        # fragments cannot be used, even when a source never received its ACK.
        for n in (1,2):
            for p in list(caches[n]):
                if p+ttl<=t:caches[n].pop(p);head_known_sink[n].pop(p,None)
        for p in list(sink):
            if p+ttl<=t:sink.pop(p)
        if tape['arrivals'][t]<arrival_p:
            births.add(t)
            if not alive[0]:terminal[t]='death'
            elif len(q)==2:terminal[t]='overflow'
            else:q.append(t)
        events=[];cursor=CONTROL_US;granted=0
        ready=[alive[n] and available(n,t)>=CONTROL_UJ for n in (0,head)]
        for n,r in zip((0,head),ready):
            if r:pay(n,CONTROL_UJ,'control_uj')
        if all(ready) and alive[0] and alive[head]:
            if observed_head!=head:source_known.clear();observed_head=head
            report=encode_report(int(head==2),head,available(0,t),[(p,p+ttl) for p in q])
            report+=bytes([source_known.get(p,0) for p in q]+[0]*(2-len(q)))
            source_free,reported=decode_report(report[:-2],int(head==2),head)
            assert source_free==available(0,t)
            source_masks={p:report[-2+i] for i,(p,_) in enumerate(reported)}
            for p,_ in sorted(reported,key=lambda z:(z[1],z[0])):
                if granted>=2 or p not in q or not alive[0] or not alive[head]:break
                cached=caches[head].get(p,0);known_sink=head_known_sink[head].get(p,0)
                plan=choose_plan(policy,source_masks.get(p,0),cached,known_sink,available(0,t),available(head,t))
                if plan is None:continue
                member,forward=plan;cs,ch,max_duration=plan_cost(policy,member,forward)
                assert cs<=available(0,t) and ch<=available(head,t)
                assert cursor+max_duration<FRAME_US
                # Local report credit minus prior public, deterministic debits
                # is the source budget for subsequent grants in this frame.
                grant=PLAN.pack(int(head==2),0,head,p,mask(member),mask(forward))
                e,s,h,gid,mm,fm=PLAN.unpack(grant)
                assert (e,s,h,gid)==(int(head==2),0,observed_head,p)
                assert bits(mm)==member and bits(fm)==forward
                granted+=1;counts['grant_frames']+=1
                before=cursor;cursor+=PLAN_US
                durations={'member_uj':0,'forward_uj':0,'feedback_uj':PLAN_US}
                if member and len(member)<N:counts['partial_windows']+=1
                if policy=='whole':caches[head][p]=0;head_known_sink[head][p]=0;sink[p]=0
                caches[head].setdefault(p,0);head_known_sink[head].setdefault(p,0)

                def ack(hop,index,bitmap):
                    nonlocal cursor
                    packet=ack_packet(p,head,bitmap);cursor+=ACK_US
                    durations['feedback_uj']+=ACK_US;counts['ack_frames']+=1
                    if tape['ack'][t,p,hop,index]>=ack_loss:
                        value=parse_ack(packet,p,head)
                        if hop==0:source_known[p]=source_known.get(p,0)|value
                        else:head_known_sink[head][p]|=value

                for i in member:
                    encoded=fragment(p,i);assert parse_fragment(encoded,p)==i
                    cursor+=DATA_US[i];durations['member_uj']+=DATA_US[i];counts['data_fragments']+=1
                    if received(t,p,0,i):
                        counts['duplicate_fragments']+=int(bool(caches[head][p]&(1<<i)))
                        caches[head][p]|=1<<i
                    if policy=='selective':ack(0,i,caches[head][p])
                if member and policy!='selective':ack(0,N,caches[head][p])
                actual_forward=tuple(i for i in forward if caches[head][p]&(1<<i))
                if policy=='whole' and caches[head][p]!=FULL:actual_forward=()
                for i in actual_forward:
                    cursor+=DATA_US[i];durations['forward_uj']+=DATA_US[i];counts['data_fragments']+=1
                    if received(t,p,1,i):
                        counts['duplicate_fragments']+=int(bool(sink.get(p,0)&(1<<i)))
                        sink[p]=sink.get(p,0)|(1<<i)
                    if policy=='selective':ack(1,i,sink.get(p,0))
                    if sink.get(p,0)==FULL and t*FRAME_US+cursor<(p+ttl)*FRAME_US:delivered.add(p)
                if actual_forward and policy!='selective':ack(1,N,sink.get(p,0))
                # Paid end-to-end receipt is transmitted even when bitmap is
                # incomplete; source releases custody ONLY after FULL receipt.
                # End receipt is at the FIXED grant-window boundary, independent
                # of hidden fragment/ACK outcomes. Both nodes keep the declared
                # RX envelope through unused positions; no silent early release.
                padding=before+max_duration-ACK_US-cursor
                assert padding>=0
                cursor+=padding;durations['feedback_uj']+=padding
                final=ack_packet(p,head,head_known_sink[head][p]);cursor+=ACK_US
                durations['feedback_uj']+=ACK_US;counts['ack_frames']+=1
                if tape['receipt'][t,p]>=ack_loss and parse_ack(final,p,head)==FULL:
                    q.remove(p)
                if head_known_sink[head][p]==FULL:
                    # Eviction depends on the HEAD's received sink ACK, never
                    # on knowledge of whether the source received its receipt.
                    caches[head].pop(p,None)
                if policy=='whole':source_known.pop(p,None)
                actual_duration=cursor-before;assert actual_duration==max_duration
                src_charge=ceil(actual_duration*.05634)+2;head_charge=ceil(actual_duration*.05634)+22
                # Categories are exact integer debits; allocate rounding and
                # processing residual to feedback to preserve conservation.
                member_charge=2*int(durations['member_uj']*.05634)
                forward_charge=2*int(durations['forward_uj']*.05634)
                counts['member_uj']+=member_charge;counts['forward_uj']+=forward_charge
                counts['feedback_uj']+=src_charge+head_charge-member_charge-forward_charge
                pay(0,src_charge,'data_uj');pay(head,head_charge,'data_uj')
                if trace:events.append(dict(pid=p,member=list(member),forward=list(actual_forward),
                    grant_hex=grant.hex(),final_receipt_hex=final.hex(),start_us=before,end_us=cursor,
                    source_uj=src_charge,head_uj=head_charge))
        for n in range(3):
            if alive[n]:
                gain=min(harvest_uj,capacity_uj-battery[n]);battery[n]+=gain;harvested[n]+=gain
            assert capacity_uj+harvested[n]==battery[n]+spent[n]
        assert len(q)<=2 and all(len(caches[n])<=2 for n in (1,2)) and granted<=2
        assert not(set(terminal)&delivered) and births==set(terminal)|delivered|set(q)
        assert alive[0] or not q
        assert counts['data_uj']==counts['member_uj']+counts['forward_uj']+counts['feedback_uj']
        if trace:records.append(dict(frame=t,head=head,observed_head=observed_head,queue=list(q),
            delivered=sorted(delivered),battery_uj=list(battery),events=events))
    losses={k:sum(v==k for v in terminal.values()) for k in ('death','overflow','stale')}
    pending=sum(p not in delivered for p in q)
    assert len(delivered)+sum(losses.values())+pending==len(births)
    assert sum(spent)==counts['background_uj']+counts['control_uj']+counts['data_uj']
    return dict(delivered=len(delivered),generated=len(births),pending_undelivered=pending,
        **losses,energy_uj=sum(spent),**counts,records=records)
