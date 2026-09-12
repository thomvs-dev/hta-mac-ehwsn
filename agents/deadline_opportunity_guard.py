"""Causal, deterministic guard on bounded-reserve deferral.

This first prototype predicts one-step geometry under the CURRENT CH only.
It is not a calibrated contact probability or a long-horizon optimal policy.
"""
import numpy as np
from agents.bounded_reserve_frontier import cluster_action as reserve_action
from agents.feasible_service_dp import cluster_action as incumbent


class MotionHistory:
    def __init__(self):
        self.previous = None

    def velocity(self, positions):
        positions = np.asarray(positions, dtype=float)
        if self.previous is None:
            return np.zeros_like(positions)
        if self.previous.shape != positions.shape:
            raise ValueError('node set changed without resetting motion history')
        return positions - self.previous

    def observe(self, positions):
        self.previous = np.asarray(positions, dtype=float).copy()


def cluster_action(base, cluster, reserve_memory, motion, table):
    """Permit reserve deferral only if omitted service passes a causal check."""
    full = incumbent(base, cluster, table)
    candidate = reserve_action(base, cluster, reserve_memory, table, 'reserve_only')
    if np.array_equal(full, candidate):
        return full
    ch = int(base.cluster_heads[cluster])
    projected = base.positions + motion.velocity(base.positions)
    total = int(candidate.sum())
    # Fixed-total optima can reallocate several packets even if totals differ
    # by one. Check EVERY omitted FIFO packet, not merely one chosen node.
    for node in np.flatnonzero(full > candidate):
        omitted = base.packet_ages[node][int(candidate[node]):int(full[node])]
        if any(age >= base.cfg.packet_ttl_rounds for age in omitted):
            return full
        distance = float(np.linalg.norm(projected[node] - projected[ch]))
        next_packet_cost = base.radio.tx(base.cfg.packet_bits, distance)
        remaining = base.energy[node] - table.member[node, total, int(candidate[node])]
        # Forecast harvest cannot fund the affordability guarantee. Future
        # idle/RX/forward costs and CH changes remain explicit limitations.
        if next_packet_cost > remaining:
            return full
    return candidate
