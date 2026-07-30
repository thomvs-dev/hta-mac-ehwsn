"""Gymnasium-style intra-cluster MAC environment for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.energy.idle_model import energy_update, idle_listening_energy
from core.energy.radio_model import RadioModel
from core.hmm.solar_hmm import HMMParameters


@dataclass(frozen=True)
class MACEnvironmentConfig:
    initial_energy_j: float
    packet_bits: int
    e_elec_j_per_bit: float
    frame_slot_budget: int
    n_max: int
    queue_max_packets: int
    max_rounds: int
    solar_scale: float
    thermal_scale: float
    bs_position_m: tuple[float, float]


class IntraClusterMACEnv:
    """Evaluate slot allocation under a frozen initial HEART-CH clustering.

    The CH decision and ST-GCN embedding enter through ``frozen_snapshot`` and
    are never trained or changed here.
    """

    def __init__(
        self,
        config: MACEnvironmentConfig,
        radio: RadioModel,
        solar: HMMParameters,
        thermal: HMMParameters,
        *,
        idle_energy_enabled: bool = True,
    ) -> None:
        self.cfg = config
        self.radio = radio
        self.solar = solar
        self.thermal = thermal
        self.idle_energy_enabled = bool(idle_energy_enabled)
        self.np_random = np.random.default_rng()
        self.round = 0

    def reset(self, *, seed: int, frozen_snapshot: dict):
        self.np_random = np.random.default_rng(seed)
        self.positions = np.asarray(
            frozen_snapshot["positions"], dtype=np.float64
        ).copy()
        self.cluster_heads = np.asarray(
            frozen_snapshot["cluster_heads"], dtype=np.int64
        ).copy()
        self.solar_states = np.asarray(
            frozen_snapshot["solar_states"], dtype=np.int64
        ).copy()
        self.thermal_states = np.asarray(
            frozen_snapshot["thermal_states"], dtype=np.int64
        ).copy()
        self.embedding = np.asarray(frozen_snapshot["stgcn_embedding"]).copy()
        self.n_nodes = self.positions.shape[0]
        self.energy = np.full(
            self.n_nodes, self.cfg.initial_energy_j, dtype=np.float64
        )
        self.alive = np.ones(self.n_nodes, dtype=bool)
        self.queue = np.ones(self.n_nodes, dtype=np.int64)
        self.previous_slots = np.zeros(self.n_nodes, dtype=np.int64)
        self.round = 0
        self.t_fnd = None
        self.total_idle_energy = 0.0
        self.total_packets = 0
        self._assign_clusters()
        return self._state(), self._info()

    def _assign_clusters(self) -> None:
        ch_positions = self.positions[self.cluster_heads]
        distances = np.linalg.norm(
            self.positions[:, None, :] - ch_positions[None, :, :], axis=2
        )
        self.cluster_of = np.argmin(distances, axis=1)
        for local, ch in enumerate(self.cluster_heads):
            self.cluster_of[ch] = local

    def static_equal_action(self) -> np.ndarray:
        action = np.zeros(self.n_nodes, dtype=np.int64)
        ch_mask = np.zeros(self.n_nodes, dtype=bool)
        ch_mask[self.cluster_heads] = True
        action[self.alive & ~ch_mask] = 1
        return action

    def _sample_harvest(self) -> np.ndarray:
        solar_raw = self.np_random.normal(
            self.solar.mean[self.solar_states],
            np.sqrt(self.solar.variance[self.solar_states]),
        )
        thermal_raw = self.np_random.normal(
            self.thermal.mean[self.thermal_states],
            np.sqrt(self.thermal.variance[self.thermal_states]),
        )
        harvested = np.maximum(0.0, solar_raw * self.cfg.solar_scale)
        harvested += np.maximum(0.0, thermal_raw * self.cfg.thermal_scale)

        uniforms = self.np_random.random(self.n_nodes)
        solar_cdf = np.cumsum(self.solar.transition[self.solar_states], axis=1)
        thermal_cdf = np.cumsum(
            self.thermal.transition[self.thermal_states], axis=1
        )
        self.solar_states = np.sum(uniforms[:, None] > solar_cdf, axis=1)
        uniforms = self.np_random.random(self.n_nodes)
        self.thermal_states = np.sum(uniforms[:, None] > thermal_cdf, axis=1)
        return harvested

    def _validate_action(self, action: np.ndarray) -> None:
        if action.shape != (self.n_nodes,):
            raise ValueError(f"action must have shape ({self.n_nodes},)")
        if np.any(action < 0) or np.any(action > self.cfg.n_max):
            raise ValueError("slot count outside [0,n_max]")
        if np.any(action[~self.alive] != 0):
            raise ValueError("dead nodes cannot receive slots")
        if np.any(action[self.cluster_heads] != 0):
            raise ValueError("CH branches are not member-slot branches")
        for cluster in range(len(self.cluster_heads)):
            members = self.cluster_of == cluster
            if int(action[members].sum()) > self.cfg.frame_slot_budget:
                raise ValueError("per-cluster frame budget exceeded")

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        self._validate_action(action)
        energy_before = self.energy.copy()
        consumed = np.zeros(self.n_nodes, dtype=np.float64)
        idle = np.zeros(self.n_nodes, dtype=np.float64)
        delivered = np.minimum(self.queue, action)

        for cluster, ch in enumerate(self.cluster_heads):
            member_mask = (
                (self.cluster_of == cluster)
                & self.alive
                & (np.arange(self.n_nodes) != ch)
            )
            active = np.flatnonzero(member_mask & (action > 0))
            frame_slots = int(action[active].sum())
            for node in active:
                distance = float(np.linalg.norm(self.positions[node] - self.positions[ch]))
                consumed[node] += self.radio.tx(
                    self.cfg.packet_bits * int(delivered[node]), distance
                )
                if self.idle_energy_enabled:
                    idle_slots = max(0, frame_slots - int(action[node]))
                    idle[node] = float(
                        idle_listening_energy(
                            idle_slots,
                            p_idle_j_per_bit_time=self.cfg.e_elec_j_per_bit,
                            slot_bit_times=self.cfg.packet_bits,
                        )
                    )
            received = int(delivered[active].sum())
            if self.alive[ch] and received:
                bits = self.cfg.packet_bits * received
                consumed[ch] += self.radio.rx(bits) + self.radio.aggregate(bits)
                distance_bs = float(
                    np.linalg.norm(
                        self.positions[ch] - np.asarray(self.cfg.bs_position_m)
                    )
                )
                consumed[ch] += self.radio.tx(self.cfg.packet_bits, distance_bs)

        consumed += idle
        harvested = self._sample_harvest()
        self.energy = energy_update(
            self.energy, consumed, harvested, self.cfg.initial_energy_j
        )
        self.alive = self.energy > 0.0
        self.queue = np.minimum(
            self.cfg.queue_max_packets,
            np.maximum(0, self.queue - delivered) + self.alive.astype(np.int64),
        )
        self.queue[~self.alive] = 0
        self.previous_slots = action.copy()
        self.round += 1
        if self.t_fnd is None and not np.all(self.alive):
            self.t_fnd = self.round
        self.total_idle_energy += float(idle.sum())
        self.total_packets += int(delivered.sum())

        trace = {
            "round": self.round,
            "energy_before": energy_before,
            "consumed": consumed,
            "harvested": harvested,
            "energy_after": self.energy.copy(),
            "idle_energy": idle,
        }
        terminated = not np.any(self.alive)
        truncated = self.round >= self.cfg.max_rounds
        info = self._info()
        info["energy_trace"] = trace
        return self._state(), 0.0, terminated, truncated, info

    def _state(self) -> np.ndarray:
        solar_rows = self.solar.transition[self.solar_states]
        thermal_rows = self.thermal.transition[self.thermal_states]
        expected = self.solar.transition @ self.solar.mean
        expected_t = self.thermal.transition @ self.thermal.mean
        second = self.solar.transition @ (
            self.solar.variance + self.solar.mean**2
        )
        second_t = self.thermal.transition @ (
            self.thermal.variance + self.thermal.mean**2
        )
        forecast = (
            expected[self.solar_states] * self.cfg.solar_scale
            + expected_t[self.thermal_states] * self.cfg.thermal_scale
        )
        variance = (
            np.maximum(0.0, second[self.solar_states] - expected[self.solar_states] ** 2)
            * self.cfg.solar_scale**2
            + np.maximum(
                0.0,
                second_t[self.thermal_states] - expected_t[self.thermal_states] ** 2,
            )
            * self.cfg.thermal_scale**2
        )
        cluster_fraction = np.bincount(
            self.cluster_of, minlength=len(self.cluster_heads)
        )[self.cluster_of] / self.n_nodes
        state = np.column_stack(
            (
                self.energy / self.cfg.initial_energy_j,
                forecast,
                variance,
                solar_rows,
                thermal_rows,
                self.queue / self.cfg.queue_max_packets,
                self.previous_slots / self.cfg.n_max,
                cluster_fraction,
            )
        )
        if state.shape != (self.n_nodes, 18):
            raise RuntimeError(f"invalid state shape {state.shape}")
        return state.astype(np.float32)

    def _info(self) -> dict:
        return {
            "round": self.round,
            "alive": int(self.alive.sum()),
            "t_fnd": self.t_fnd,
            "total_idle_energy_j": self.total_idle_energy,
            "packets_delivered": self.total_packets,
            "cluster_heads": self.cluster_heads.copy(),
        }
