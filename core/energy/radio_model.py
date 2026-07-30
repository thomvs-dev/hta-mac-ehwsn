"""Exact, configuration-driven HEART-CH first-order radio model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RadioModel:
    e_elec_j_per_bit: float
    eps_fs_j_per_bit_m2: float
    eps_mp_j_per_bit_m4: float
    e_da_j_per_bit: float
    d0_m: float

    def tx(self, bits: int, distance_m: float) -> float:
        if bits < 0 or distance_m < 0:
            raise ValueError("bits and distance must be non-negative")
        amplifier = (
            self.eps_fs_j_per_bit_m2 * distance_m**2
            if distance_m < self.d0_m
            else self.eps_mp_j_per_bit_m4 * distance_m**4
        )
        return bits * (self.e_elec_j_per_bit + amplifier)

    def rx(self, bits: int) -> float:
        if bits < 0:
            raise ValueError("bits must be non-negative")
        return bits * self.e_elec_j_per_bit

    def aggregate(self, bits: int) -> float:
        if bits < 0:
            raise ValueError("bits must be non-negative")
        return bits * self.e_da_j_per_bit

