"""Regression test for inherited permanent-death harvest semantics."""

from __future__ import annotations

from envs import IntraClusterMACEnv
from experiments.calibrate_reward_scales import _setup


def test_dead_node_receives_no_harvest_or_hmm_transition():
    snapshot, solar, thermal, radio, config = _setup()
    env = IntraClusterMACEnv(config, radio, solar, thermal)
    env.reset(seed=123, frozen_snapshot=snapshot)
    env.alive[7] = False
    env.energy[7] = 0.0
    solar_state = int(env.solar_states[7])
    thermal_state = int(env.thermal_states[7])

    harvested = env._sample_harvest()

    assert harvested[7] == 0.0
    assert int(env.solar_states[7]) == solar_state
    assert int(env.thermal_states[7]) == thermal_state

def test_packet_accounting_counts_generation_and_death_drops():
    snapshot, solar, thermal, radio, config = _setup()
    env = IntraClusterMACEnv(config, radio, solar, thermal)
    env.reset(seed=124, frozen_snapshot=snapshot)
    initial_generated = env.total_packets_generated
    env.packet_ages[7] = [0, 1, 2]
    env.alive[7] = False

    delivered = env.queue * 0
    delivered[7] = 1
    env._update_queues(delivered)

    assert env.dropped_death_packets == 2
    assert env.packet_ages[7] == []
    assert env.total_packets_generated == initial_generated + env.n_nodes - 1
    assert env._info()["packets_generated"] == env.total_packets_generated
