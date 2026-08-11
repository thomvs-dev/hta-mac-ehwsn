"""Step 3 wrappers: role-separated energy evidence and CH-risk reward term."""

from __future__ import annotations

import numpy as np

from agents.ch_depletion_risk import ch_depletion_risk, validate_ch_risk_config
from envs.dynamic_cluster_training_env import DynamicClusterTrainingEnv
from envs.scheduled_mac_env import ScheduledIntraClusterMACEnv


_ACTIVE_RISK_CONFIG = None


def configure_step3_risk(config: dict) -> None:
    global _ACTIVE_RISK_CONFIG
    _ACTIVE_RISK_CONFIG = validate_ch_risk_config(dict(config))


class RoleSeparatedScheduledMACEnv(ScheduledIntraClusterMACEnv):
    """Add exact role-energy components without changing environment dynamics."""

    def step(self, action):
        action = np.asarray(action, dtype=np.int64)
        heads = self.cluster_heads.copy()
        cluster_of = self.cluster_of.copy()
        alive = self.alive.copy()
        positions = self.positions.copy()
        state, reward, terminated, truncated, info = super().step(action)
        delivered = np.asarray(info["delivered_packets_per_node"], dtype=np.int64)
        member_tx = np.zeros(self.n_nodes, dtype=np.float64)
        ch_rx = np.zeros(self.n_nodes, dtype=np.float64)
        ch_aggregate = np.zeros(self.n_nodes, dtype=np.float64)
        ch_tx_bs = np.zeros(self.n_nodes, dtype=np.float64)
        node_ids = np.arange(self.n_nodes)
        for cluster, ch_value in enumerate(heads):
            ch = int(ch_value)
            members = (cluster_of == cluster) & alive & (node_ids != ch)
            active = np.flatnonzero(members & (action > 0))
            for node in active:
                distance = float(np.linalg.norm(positions[node] - positions[ch]))
                member_tx[node] += self.radio.tx(
                    self.cfg.packet_bits * int(delivered[node]), distance
                )
            received = int(delivered[active].sum())
            if alive[ch] and received:
                bits = self.cfg.packet_bits * received
                ch_rx[ch] += self.radio.rx(bits)
                ch_aggregate[ch] += self.radio.aggregate(bits)
                distance_bs = float(
                    np.linalg.norm(
                        positions[ch] - np.asarray(self.cfg.bs_position_m)
                    )
                )
                ch_tx_bs[ch] += self.radio.tx(self.cfg.packet_bits, distance_bs)
        idle = np.asarray(info["energy_trace"]["idle_energy"], dtype=np.float64)
        reconstructed = member_tx + ch_rx + ch_aggregate + ch_tx_bs + idle
        consumed = np.asarray(info["energy_trace"]["consumed"], dtype=np.float64)
        if not np.allclose(reconstructed, consumed, rtol=0.0, atol=1e-15):
            raise RuntimeError("role-separated energy does not reconstruct consumption")
        info["energy_trace"]["role_energy"] = {
            "member_tx": member_tx,
            "ch_rx": ch_rx,
            "ch_aggregate": ch_aggregate,
            "ch_tx_bs": ch_tx_bs,
            "idle": idle,
            "reconstructed_consumed": reconstructed,
        }
        return state, reward, terminated, truncated, info


class Step3DynamicClusterTrainingEnv(DynamicClusterTrainingEnv):
    """Append an observable, scheduled-CH-conditioned depletion-risk term."""

    def step(self, member_action):
        if _ACTIVE_RISK_CONFIG is None:
            raise RuntimeError("Step 3 risk config has not been installed")
        current_ch = int(self.ch)
        members = self.members.copy()
        action = np.asarray(member_action, dtype=np.int64)
        intended = int(np.minimum(self.base.queue[members], action[members]).sum())
        physical_state = self.base._state()
        reserve_fraction = float(
            self.base.energy[current_ch] / self.base.cfg.initial_energy_j
        )
        forecast_harvest_j = float(physical_state[current_ch, 1])
        distance_to_bs_m = float(
            np.linalg.norm(
                self.base.positions[current_ch]
                - np.asarray(self.base.cfg.bs_position_m)
            )
        )
        risk = ch_depletion_risk(
            _ACTIVE_RISK_CONFIG,
            reserve_fraction=reserve_fraction,
            forecast_harvest_j=forecast_harvest_j,
            distance_to_bs_m=distance_to_bs_m,
            intended_delivered_packets=intended,
            frame_slot_budget=self.base.cfg.frame_slot_budget,
        )
        observation, mask, done, info = super().step(member_action)
        info["reward_raw_terms"]["ch_depletion_risk"] = risk["raw_penalty"]
        role = info["energy_trace"]["role_energy"]
        risk["target_ch_role_energy_j"] = {
            name: float(values[current_ch]) for name, values in role.items()
        }
        risk["target_ch_energy_after_j"] = float(self.base.energy[current_ch])
        info["ch_depletion_risk"] = risk
        return observation, mask, done, info
