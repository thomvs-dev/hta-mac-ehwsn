"""Gymnasium-style intra-cluster MAC environment for HTA-MAC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.energy.idle_model import energy_update, idle_listening_energy
from core.energy.radio_model import RadioModel
from core.hmm.rectified_moments import next_rectified_statistics
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
    control_packet_bits: int = 100
    packet_ttl_rounds: int = 3
    idle_slot_bit_times: int = 4000


class IntraClusterMACEnv:
    """Evaluate slot allocation under externally frozen HEART-CH decisions."""

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
        self.packet_ages: list[list[int]] = [[0] for _ in range(self.n_nodes)]
        self.queue = np.ones(self.n_nodes, dtype=np.int64)
        self.previous_slots = np.zeros(self.n_nodes, dtype=np.int64)
        self.round = 0
        self.t_fnd = None
        self.total_idle_energy = 0.0
        self.total_packets = 0
        self.dropped_stale_packets = 0
        self.dropped_overflow_packets = 0
        self.max_backlog_observed = 1
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
        """Fair round-robin equal-slot baseline under a scarce frame budget."""
        action = np.zeros(self.n_nodes, dtype=np.int64)
        for cluster, ch in enumerate(self.cluster_heads):
            if not self.alive[ch]:
                continue
            members = np.flatnonzero(
                (self.cluster_of == cluster)
                & self.alive
                & (np.arange(self.n_nodes) != ch)
            )
            if not len(members):
                continue
            offset = self.round % len(members)
            ordered = np.roll(members, -offset)
            selected = ordered[: self.cfg.frame_slot_budget]
            action[selected] = 1
        return action

    def _sample_harvest(self) -> np.ndarray:
        harvested = np.zeros(self.n_nodes, dtype=np.float64)
        alive = np.flatnonzero(self.alive)
        if not len(alive):
            return harvested

        solar_state = self.solar_states[alive]
        thermal_state = self.thermal_states[alive]
        solar_raw = self.np_random.normal(
            self.solar.mean[solar_state],
            np.sqrt(self.solar.variance[solar_state]),
        )
        thermal_raw = self.np_random.normal(
            self.thermal.mean[thermal_state],
            np.sqrt(self.thermal.variance[thermal_state]),
        )
        harvested[alive] = np.maximum(
            0.0, solar_raw * self.cfg.solar_scale
        )
        harvested[alive] += np.maximum(
            0.0, thermal_raw * self.cfg.thermal_scale
        )

        uniforms = self.np_random.random(len(alive))
        solar_cdf = np.cumsum(
            self.solar.transition[solar_state], axis=1
        )
        self.solar_states[alive] = np.sum(
            uniforms[:, None] > solar_cdf, axis=1
        )
        uniforms = self.np_random.random(len(alive))
        thermal_cdf = np.cumsum(
            self.thermal.transition[thermal_state], axis=1
        )
        self.thermal_states[alive] = np.sum(
            uniforms[:, None] > thermal_cdf, axis=1
        )
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
        for cluster, ch in enumerate(self.cluster_heads):
            members = self.cluster_of == cluster
            if not self.alive[ch] and np.any(action[members] != 0):
                raise ValueError(
                    "members cannot receive slots when their scheduled CH is dead"
                )
            if int(action[members].sum()) > self.cfg.frame_slot_budget:
                raise ValueError("per-cluster frame budget exceeded")

    def _update_queues(self, delivered: np.ndarray) -> None:
        for node in range(self.n_nodes):
            if not self.alive[node]:
                self.packet_ages[node] = []
                continue
            served = min(int(delivered[node]), len(self.packet_ages[node]))
            if served:
                del self.packet_ages[node][:served]
            aged = [age + 1 for age in self.packet_ages[node]]
            retained = [
                age for age in aged if age <= self.cfg.packet_ttl_rounds
            ]
            self.dropped_stale_packets += len(aged) - len(retained)
            retained.append(0)
            if len(retained) > self.cfg.queue_max_packets:
                overflow = len(retained) - self.cfg.queue_max_packets
                self.dropped_overflow_packets += overflow
                retained = retained[overflow:]
            self.packet_ages[node] = retained
        self.queue = np.asarray(
            [len(packets) for packets in self.packet_ages], dtype=np.int64
        )
        self.max_backlog_observed = max(
            self.max_backlog_observed, int(self.queue.max(initial=0))
        )

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
                distance = float(
                    np.linalg.norm(self.positions[node] - self.positions[ch])
                )
                consumed[node] += self.radio.tx(
                    self.cfg.packet_bits * int(delivered[node]), distance
                )
                if self.idle_energy_enabled:
                    idle_slots = max(0, frame_slots - int(action[node]))
                    idle[node] = float(
                        idle_listening_energy(
                            idle_slots,
                            p_idle_j_per_bit_time=self.cfg.e_elec_j_per_bit,
                            slot_bit_times=self.cfg.idle_slot_bit_times,
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
        self._update_queues(delivered)
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
        info["delivered_packets_per_node"] = delivered.copy()
        return self._state(), 0.0, terminated, truncated, info

    def _state(self) -> np.ndarray:
        solar_transition = self.solar.transition[self.solar_states]
        thermal_transition = self.thermal.transition[self.thermal_states]
        solar_mean, solar_variance = next_rectified_statistics(
            self.solar.transition,
            self.solar.mean,
            self.solar.variance,
            self.cfg.solar_scale,
        )
        thermal_mean, thermal_variance = next_rectified_statistics(
            self.thermal.transition,
            self.thermal.mean,
            self.thermal.variance,
            self.cfg.thermal_scale,
        )
        forecast = (
            solar_mean[self.solar_states]
            + thermal_mean[self.thermal_states]
        )
        variance = (
            solar_variance[self.solar_states]
            + thermal_variance[self.thermal_states]
        )
        cluster_fraction = np.bincount(
            self.cluster_of, minlength=len(self.cluster_heads)
        )[self.cluster_of] / self.n_nodes
        state = np.column_stack(
            (
                self.energy / self.cfg.initial_energy_j,
                forecast,
                variance,
                solar_transition,
                thermal_transition,
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
            "dropped_stale_packets": self.dropped_stale_packets,
            "dropped_overflow_packets": self.dropped_overflow_packets,
            "max_backlog_observed": self.max_backlog_observed,
            "cluster_heads": self.cluster_heads.copy(),
        }
