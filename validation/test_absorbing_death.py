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
