"""Controlled environment variant for validating the idle term in isolation."""

from __future__ import annotations

from .intra_cluster_mac_env import IntraClusterMACEnv


class _MemberEnergyRadioProxy:
    """Keep member transmission physics while holding CH Rx/DA load exogenous."""

    def __init__(self, radio):
        self._radio = radio

    def tx(self, bits, distance_m):
        return self._radio.tx(bits, distance_m)

    def rx(self, bits):
        return 0.0

    def aggregate(self, bits):
        return 0.0


class IdleIsolationMACEnv(IntraClusterMACEnv):
    """Phase 1 diagnostic only; not valid for protocol performance reporting."""

    def __init__(self, config, radio, solar, thermal, **kwargs):
        super().__init__(
            config,
            _MemberEnergyRadioProxy(radio),
            solar,
            thermal,
            **kwargs,
        )
