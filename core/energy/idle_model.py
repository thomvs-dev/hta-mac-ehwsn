"""Explicit idle-listening energy accounting for HTA-MAC."""

from __future__ import annotations

import numpy as np


def idle_listening_energy(
    idle_slots,
    *,
    p_idle_j_per_bit_time: float,
    slot_bit_times: int,
):
    """Return energy spent awake in unused TDMA slots.

    HEART-CH supplies ``E_elec`` as the only receiver/idle electronics
    constant. A slot lasts one packet transmission, expressed in bit-times.
    """
    slots = np.asarray(idle_slots, dtype=np.float64)
    if np.any(slots < 0) or p_idle_j_per_bit_time < 0 or slot_bit_times < 0:
        raise ValueError("idle-energy inputs must be non-negative")
    return slots * float(p_idle_j_per_bit_time) * int(slot_bit_times)


def energy_update(energy, consumed, harvested, capacity):
    """Apply E(t+1)=max(0,min(Emax,E(t)-C(t)+H(t)))."""
    e = np.asarray(energy, dtype=np.float64)
    c = np.asarray(consumed, dtype=np.float64)
    h = np.asarray(harvested, dtype=np.float64)
    if np.any(c < 0) or np.any(h < 0) or capacity < 0:
        raise ValueError("energy terms and capacity must be non-negative")
    return np.maximum(0.0, np.minimum(float(capacity), e - c + h))
