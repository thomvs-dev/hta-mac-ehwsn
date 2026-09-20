"""Integer-time, per-cluster radio schedule; separate from frozen simulators.

This layer returns events and required radio energy, never delivery credits.
Typical-device parameters and protocol assumptions are declared in the plan.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Timing:
    frame_us: int = 1_000_000
    max_members: int = 99
    budget: int = 24
    payload_bytes: int = 500
    guard_us: int = 400
    wake_us: int = 192
    deep_wake_us: int = 1192  # Typical crystal startup (1000) + warm PLL (192).
    volts: float = 3.
    rx_amp: float = .0188
    tx_amp: float = .0174
    idle_amp: float = .000426  # Crystal-on IDLE, not deep sleep.
    deep_amp: float = .000020  # Regulator-on power-down, crystal off.
    extra_wake_j: float = 1e-5  # Assumption, not measured CC2420 energy.

    def __post_init__(self):
        for name in ('frame_us', 'max_members', 'budget', 'payload_bytes', 'guard_us', 'wake_us', 'deep_wake_us'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'invalid {name}')
        for name in ('volts', 'rx_amp', 'tx_amp', 'idle_amp', 'deep_amp', 'extra_wake_j'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'invalid {name}')
        if not self.volts or self.rx_amp < self.idle_amp or self.deep_amp > self.idle_amp or self.deep_wake_us < self.wake_us:
            raise ValueError('invalid active/idle powers')
        # Assumed 40 ppm per clock plus 20 us synchronization uncertainty.
        if self.guard_us < self.wake_us + math.ceil(80e-6*self.frame_us) + 20:
            raise ValueError('guard does not cover declared transition/drift budget')


@dataclass(frozen=True)
class Event:
    node: int
    start_us: int
    end_us: int
    state: str
    phase: str


def wire_fragments(payload_bytes):
    """127-byte PSDU - 11-byte MAC - 6-byte fragment header = 110 data bytes."""
    if type(payload_bytes) is not int or payload_bytes <= 0:
        raise ValueError('invalid payload')
    return [min(110, payload_bytes-i)+23 for i in range(0, payload_bytes, 110)]


def compile_frame(members, head, allocation, timing=Timing(), grant_received=None):
    """Members arrive in declared spatial order; grants do not reveal future state.

    grant_received is an execution outcome, not an allocator observation.
    Reserved slots of a member missing its grant are never reallocated.
    """
    members = list(members)
    if len(members) > timing.max_members or len(set(members+[head])) != len(members)+1:
        raise ValueError('duplicate identities or excessive membership')
    if set(allocation) != set(members):
        raise ValueError('allocation must cover exactly the members')
    if any(type(k) is not int or k < 0 for k in allocation.values()):
        raise ValueError('nonnegative integer allocations required')
    if sum(allocation.values()) > timing.budget:
        raise ValueError('cluster data budget exceeded')
    granted = {n: True for n in members} if grant_received is None else dict(grant_received)
    if set(granted) != set(members) or any(type(x) is not bool for x in granted.values()):
        raise ValueError('explicit Boolean grant outcomes required')
    events = []
    cursor = timing.wake_us

    def packet(sender, receivers, wire_bytes, phase, sent=True):
        nonlocal cursor
        start = cursor+timing.guard_us
        end = start+wire_bytes*32  # 250 kbit/s => 32 us per byte.
        for node, state in ([(sender, 'tx')] if sent else []) + [(n, 'rx') for n in receivers]:
            events.extend((Event(node, cursor, start, 'guard', phase),
                           Event(node, start, end, state, phase),
                           Event(node, end, end+timing.guard_us, 'guard', phase)))
        cursor = end+timing.guard_us

    control_slot = 33*32 + 2*timing.guard_us  # 16 payload + 11 MAC + 6 PHY.
    for node in members:
        packet(node, [head], 33, 'report')
    cursor = timing.wake_us + timing.max_members*control_slot
    for node in members:
        packet(head, [node], 33, 'grant')
    cursor = timing.wake_us + 2*timing.max_members*control_slot
    fragments = wire_fragments(timing.payload_bytes)
    packet_slot = sum(n*32+2*timing.guard_us for n in fragments)
    data_start = cursor
    for node in members:
        for _ in range(allocation[node]):
            for size in fragments:
                packet(node, [head], size, 'data', sent=granted[node])
    cursor = data_start+timing.budget*packet_slot
    if sum(allocation.values()):
        # Conservative reserved aggregate/ACK cost even after missing grants.
        for size in fragments:
            packet(head, [], size, 'forward')  # Mains-powered sink excluded.
        packet(head, [n for n in members if allocation[n] and granted[n]],
               17+8+math.ceil(timing.budget/8), 'block_ack')
    else:
        cursor += packet_slot + (17+8+math.ceil(timing.budget/8))*32+2*timing.guard_us
    if cursor > timing.frame_us:
        raise ValueError('reserved protocol exceeds fixed frame duration')
    for node in members+[head]:
        required = sorted((e for e in events if e.node == node), key=lambda e: e.start_us)
        if any(a.end_us > b.start_us for a, b in zip(required, required[1:])):
            raise ValueError('overlapping required radio states')
    return events


def radio_plan(node, events, timing=Timing(), mode='optimal'):
    """Minimize additive gap energy conditional on fixed required events.

    `window` stays awake between first/last events. `every_gap` uses warm sleep;
    `deep_every_gap` prefers deep sleep whenever it fits. `optimal` minimizes
    interior-gap energy under the fixed warm initial/tail convention. Frame tails
    use warm IDLE, and every activation is charged, including the first.
    """
    if mode not in ('window', 'every_gap', 'deep_every_gap', 'optimal'):
        raise ValueError('unknown radio policy')
    required = sorted((e for e in events if e.node == node), key=lambda e: e.start_us)
    power = {'idle': timing.volts*timing.idle_amp, 'rx': timing.volts*timing.rx_amp,
             'tx': timing.volts*timing.tx_amp, 'guard': timing.volts*timing.rx_amp,
             'wake': timing.volts*timing.rx_amp, 'listen': timing.volts*timing.rx_amp,
             'deep': timing.volts*timing.deep_amp, 'wake_deep': timing.volts*timing.rx_amp}
    if any(e.start_us < timing.wake_us or e.end_us > timing.frame_us or e.end_us <= e.start_us
           or e.state not in ('tx', 'rx', 'guard') for e in required):
        raise ValueError('invalid required event')
    if any(a.end_us > b.start_us for a, b in zip(required, required[1:])):
        raise ValueError('overlapping required events')
    segments = []
    wakes = 0

    def add(start, end, state):
        if end > start:
            segments.append(dict(start_us=start, end_us=end, state=state,
                                 energy_j=(end-start)*1e-6*power[state]))

    cursor = 0
    for i, event in enumerate(required):
        gap = event.start_us-cursor
        choices = [('listen', 0, gap*power['listen']*1e-6)]
        for state, duration in [('idle', timing.wake_us), ('deep', timing.deep_wake_us)]:
            if gap >= duration:
                cost = ((gap-duration)*power[state]+duration*power['wake'])*1e-6+timing.extra_wake_j
                choices.append((state, duration, cost))
        if i == 0:
            # All frame boundaries explicitly start/end in crystal-on IDLE.
            chosen = next((c for c in choices if c[0] == 'idle'), None)
            if chosen is None:
                raise ValueError('initial startup does not fit')
        elif mode == 'optimal':
            chosen = min(choices, key=lambda c: c[2])  # Stable ties retain awake.
        elif mode == 'deep_every_gap':
            chosen = choices[-1]
        elif mode == 'every_gap':
            chosen = next((c for c in choices if c[0] == 'idle'), choices[0])
        else:
            chosen = choices[0]
        state, duration, _ = chosen
        if duration:
            add(cursor, event.start_us-duration, state)
            add(event.start_us-duration, event.start_us, 'wake_deep' if state == 'deep' else 'wake')
            wakes += 1
        else:
            add(cursor, event.start_us, 'listen')
        add(event.start_us, event.end_us, event.state)
        cursor = event.end_us
    add(cursor, timing.frame_us, 'idle')
    return dict(node=node, mode=mode, segments=segments, wakes=wakes,
                switching_energy_j=wakes*timing.extra_wake_j,
                energy_j=sum(s['energy_j'] for s in segments)+wakes*timing.extra_wake_j)


def compile_radio_model(members, head, allocation, timing=Timing(), mode='optimal', grant_received=None):
    members = list(members)
    events = compile_frame(members, head, allocation, timing, grant_received)
    plans = {n: radio_plan(n, events, timing, mode) for n in list(members)+[head]}
    return dict(events=events, plans=plans, energy_j=sum(p['energy_j'] for p in plans.values()),
                frame_us=timing.frame_us, delivery_not_evaluated=True)
