"""Shared-head integration screen. Synthetic topology, not full HEART-CH."""
from math import ceil
import struct
import numpy as np
from core.fragment_repair_fixed import (
    HEADER, ACK, PLAN, CHUNKS, N, FULL, FRAME_US, OVERHEAD, DATA_US,
    ACK_US, PLAN_US, wire_time, bits, mask, choose_plan, plan_cost,
)

REPORT = struct.Struct('>IHHIBIIII BB')
SYNC_US = 1192 + wire_time(33)
REPORT_US = wire_time(REPORT.size + OVERHEAD)


def fragment(source, birth, index):
    if not 0 <= index < N:
        raise ValueError('index')
    return HEADER.pack(birth, source, index, N) + bytes([index]) * CHUNKS[index]


def parse_fragment(packet, source, birth):
    if len(packet) < HEADER.size:
        raise ValueError('length')
    p, s, i, count = HEADER.unpack(packet[:HEADER.size])
    if (p, s, count) != (birth, source, N) or not 0 <= i < N:
        raise ValueError('identity')
    if len(packet) != HEADER.size + CHUNKS[i]:
        raise ValueError('length')
    return i


def parse_ack(packet, source, birth, head):
    if len(packet) != ACK.size:
        raise ValueError('length')
    p, s, h, value = ACK.unpack(packet)
    if (p, s, h) != (birth, source, head) or value & ~FULL:
        raise ValueError('identity')
    return value


def make_tape(seed, horizon, sources):
    # Per-source streams preserve physical pairing across policies and sizes.
    return [dict(arrivals=(r := np.random.default_rng(np.random.SeedSequence([seed, s]))).random(horizon),
                 data=r.random((horizon, horizon, 2, N)),
                 ack=r.random((horizon, horizon, 2, N+1)),
                 receipt=r.random((horizon, horizon)), burst=r.random((horizon, 2)))
            for s in range(sources)]


def evaluate(tapes, policy, *, capacity_uj=16000, harvest_uj=0,
             ttl=2, loss='independent', ack_loss=.1, arrival_p=.35,
             head_change=False, trace=False):
    if policy not in ('whole', 'selective', 'block') or loss not in ('clean', 'independent', 'burst'):
        raise ValueError('mode')
    m = len(tapes); horizon = len(tapes[0]['arrivals']); nodes = m+2
    battery = [capacity_uj]*nodes; spent = [0]*nodes; gained = [0]*nodes
    alive = [capacity_uj > 0]*nodes; queues = [[] for _ in range(m)]
    caches = {m: {}, m+1: {}}; sink_known = {m: {}, m+1: {}}
    source_known = [{} for _ in range(m)]; observed = [None]*m
    births = set(); delivered = set(); terminal = {}; sink = {}; records = []
    count = dict(control_uj=0, data_uj=0, background_uj=0,
                 member_uj=0, forward_uj=0, feedback_uj=0,
                 grant_frames=0, data_fragments=0, ack_frames=0,
                 buffer_rejections=0, airtime_rejections=0, max_cache=0, max_grants=0,
                 max_frame_us=0)

    def pay(n, cost, category):
        assert isinstance(cost, int) and 0 <= cost <= battery[n]
        battery[n] -= cost; spent[n] += cost; count[category] += cost
        if battery[n] == 0:
            alive[n] = False
            if n < m:
                for p in queues[n]:
                    if p not in delivered: terminal[p] = 'death'
                queues[n].clear()
            else:
                caches[n].clear(); sink_known[n].clear()

    def free(n, t):
        return max(0, battery[n]-60*(horizon-t-1)) if alive[n] else 0

    for t in range(horizon):
        head = m + int(head_change and t >= horizon//2)
        for n in range(nodes):
            if alive[n]: pay(n, min(60, battery[n]), 'background_uj')
        for s in range(m):
            for p in list(queues[s]):
                if p[1]+ttl <= t:
                    queues[s].remove(p)
                    if p not in delivered: terminal[p] = 'stale'
            for p in list(source_known[s]):
                if p[1]+ttl <= t: source_known[s].pop(p)
        for mapping in [*caches.values(), *sink_known.values(), sink]:
            for p in list(mapping):
                if p[1]+ttl <= t: mapping.pop(p)
        for s, tape in enumerate(tapes):
            if tape['arrivals'][t] < arrival_p:
                p = (s, t); births.add(p)
                if not alive[s]: terminal[p] = 'death'
                elif len(queues[s]) == 2: terminal[p] = 'overflow'
                else: queues[s].append(p)
        cursor = 0; grants = 0; per_source = [0]*m; reports = []; credits = {}; events = []
        sync_cost = ceil(SYNC_US*.05634)+10
        if alive[head] and free(head, t) >= sync_cost:
            pay(head, sync_cost, 'control_uj'); cursor += SYNC_US
            for s in range(m):
                if free(s, t) < sync_cost: continue
                pay(s, sync_cost, 'control_uj')
                if observed[s] != head:
                    source_known[s].clear(); observed[s] = head
                rc = ceil(REPORT_US*.05634)+10
                # Paid reliable report slot, serialized on the common radio.
                if not alive[s] or not alive[head] or min(free(s,t),free(head,t)) < rc: continue
                if cursor+REPORT_US > FRAME_US: continue
                pay(s, rc, 'control_uj'); pay(head, rc, 'control_uj'); cursor += REPORT_US
                entries = [(p[1], p[1]+ttl) for p in queues[s]]
                entries += [(2**32-1, 2**32-1)]*(2-len(entries))
                masks = [source_known[s].get(p,0) for p in queues[s]]+[0]*(2-len(queues[s]))
                wire = REPORT.pack(t,s,head,free(s,t),len(queues[s]),*sum((list(e) for e in entries),[]),*masks)
                epoch,src,h,credit,k,p0,d0,p1,d1,b0,b1 = REPORT.unpack(wire)
                assert (epoch,src,h)==(t,s,head)
                credits[s] = credit
                reports += [(d,(s-t)%m,(s,p),bm) for p,d,bm in [(p0,d0,b0),(p1,d1,b1)][:k]]
        for deadline, tie, p, reported_mask in sorted(reports):
            s, birth = p; tape = tapes[s]
            if grants >= 24: break
            if per_source[s] >= 2 or not alive[s] or not alive[head] or p not in queues[s]: continue
            cached = caches[head].get(p,0); known = sink_known[head].get(p,0)
            if p not in caches[head] and len(caches[head]) >= 2:
                count['buffer_rejections'] += 1; continue
            plan = choose_plan(policy,reported_mask,cached,known,credits[s],free(head,t))
            if plan is None: continue
            member,forward = plan; cs,ch,duration = plan_cost(policy,member,forward)
            if cursor+duration > FRAME_US:
                count['airtime_rejections'] += 1; continue
            assert cs <= free(s,t) and ch <= free(head,t)
            grant = PLAN.pack(t,s,head,birth,mask(member),mask(forward))
            assert PLAN.unpack(grant)==(t,s,head,birth,mask(member),mask(forward))
            start = cursor; cursor += PLAN_US; member_time = 0; forward_time = 0
            grants += 1; per_source[s] += 1; count['grant_frames'] += 1
            if policy == 'whole':
                caches[head][p] = 0; sink_known[head][p] = 0; sink[p] = 0
            # Receipt-only transactions need no payload buffer.
            if known != FULL or member: caches[head].setdefault(p,0)
            sink_known[head].setdefault(p,0)
            count['max_cache'] = max(count['max_cache'],len(caches[head]))

            def received(hop, i):
                return loss=='clean' or (tape['burst'][t,hop]<.9 if loss=='burst' else tape['data'][t,birth,hop,i]<.9)

            def acknowledge(hop, i, value):
                nonlocal cursor
                packet = ACK.pack(birth,s,head,value); cursor += ACK_US; count['ack_frames'] += 1
                if tape['ack'][t,birth,hop,i] >= ack_loss:
                    value = parse_ack(packet,s,birth,head)
                    if hop == 0: source_known[s][p] = source_known[s].get(p,0)|value
                    else: sink_known[head][p] |= value

            for i in member:
                assert parse_fragment(fragment(s,birth,i),s,birth)==i
                cursor += DATA_US[i]; member_time += DATA_US[i]; count['data_fragments'] += 1
                if received(0,i): caches[head][p] |= 1<<i
                if policy=='selective': acknowledge(0,i,caches[head][p])
            if member and policy!='selective': acknowledge(0,N,caches[head][p])
            actual = tuple(i for i in forward if caches[head].get(p,0)&(1<<i))
            if policy=='whole' and caches[head].get(p,0)!=FULL: actual=()
            for i in actual:
                cursor += DATA_US[i]; forward_time += DATA_US[i]; count['data_fragments'] += 1
                if received(1,i): sink[p] = sink.get(p,0)|(1<<i)
                if policy=='selective': acknowledge(1,i,sink.get(p,0))
                if sink.get(p,0)==FULL and t*FRAME_US+cursor<deadline*FRAME_US: delivered.add(p)
            if actual and policy!='selective': acknowledge(1,N,sink.get(p,0))
            assert cursor <= start+duration-ACK_US
            cursor = start+duration
            receipt = ACK.pack(birth,s,head,sink_known[head][p]); count['ack_frames'] += 1
            if tape['receipt'][t,birth] >= ack_loss and parse_ack(receipt,s,birth,head)==FULL:
                queues[s].remove(p)
            if sink_known[head][p]==FULL: caches[head].pop(p,None)
            if policy=='whole': source_known[s].pop(p,None)
            mc,fc = 2*int(member_time*.05634),2*int(forward_time*.05634)
            count['member_uj'] += mc; count['forward_uj'] += fc; count['feedback_uj'] += cs+ch-mc-fc
            pay(s,cs,'data_uj'); pay(head,ch,'data_uj'); credits[s] -= cs
            if trace: events.append(dict(source=s,birth=birth,start_us=start,end_us=cursor,source_uj=cs,head_uj=ch))
        for n in range(nodes):
            if alive[n]:
                gain = min(harvest_uj,capacity_uj-battery[n]); battery[n] += gain; gained[n] += gain
            assert capacity_uj+gained[n] == battery[n]+spent[n]
        pending = {p for q in queues for p in q}
        assert births == set(terminal)|delivered|pending and not set(terminal)&delivered
        assert all(len(q)<=2 for q in queues) and all(len(c)<=2 for c in caches.values())
        assert grants<=24 and max(per_source)<=2 and cursor<=FRAME_US
        count['max_grants']=max(count['max_grants'],grants); count['max_frame_us']=max(count['max_frame_us'],cursor)
        if trace: records.append(dict(frame=t,head=head,events=events,battery_uj=list(battery),queues=[list(q) for q in queues]))
    losses={k:sum(v==k for v in terminal.values()) for k in ('death','overflow','stale')}
    pending=sum(p not in delivered for q in queues for p in q)
    assert len(delivered)+sum(losses.values())+pending==len(births)
    assert count['data_uj']==count['member_uj']+count['forward_uj']+count['feedback_uj']
    assert sum(spent)==count['data_uj']+count['control_uj']+count['background_uj']
    return dict(delivered=len(delivered),generated=len(births),pending_undelivered=pending,
                energy_uj=sum(spent),**losses,**count,records=records)
