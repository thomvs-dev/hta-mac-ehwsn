import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agents.marginal_utility_teacher import (
    MarginalTeacherConfig,
    marginal_candidates,
    marginal_utility_action,
)
from experiments.run_v2_phase_b_teacher_calibration import expand_candidates, load_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "v2_phase_b_teacher_calibration_20260901.json"


class Radio:
    def tx(self, bits, distance): return bits * (1e-7 + distance * 1e-10)
    def rx(self, bits): return bits * 5e-8
    def aggregate(self, bits): return bits * 5e-9


class Base:
    def __init__(self):
        self.n_nodes = 5
        self.cfg = SimpleNamespace(
            initial_energy_j=1.0, packet_bits=1000, packet_ttl_rounds=3,
            queue_max_packets=5, bs_position_m=(10.0, 10.0),
        )
        self.positions = np.asarray([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]])
        self.energy = np.asarray([1.0, 0.9, 0.8, 0.7, 0.6])
        self.packet_ages = [[0], [3, 1], [1, 0], [2], [0]]
        self.queue = np.asarray([1, 2, 2, 1, 1])
        self.radio = Radio()
    def _state(self):
        state = np.zeros((5, 18), dtype=np.float64); state[:, 1] = np.linspace(0.1, 0.5, 5); return state


class Env:
    def __init__(self):
        self.base = Base(); self.ch = 0
        self.step3_episode_offered_per_node = np.asarray([0, 4, 4, 4, 4])
        self.step3_episode_delivered_per_node = np.asarray([0, 1, 3, 2, 4])


def config():
    return MarginalTeacherConfig.from_payload({
        "energy_weight": 2, "stale_weight": 4, "fairness_weight": 3,
        "age_weight": .25, "queue_weight": .1, "residual_weight": .2,
        "harvest_weight": .1, "diminishing_weight": .15,
        "member_risk_weight": .5, "ch_risk_weight": .5,
        "member_reserve_floor": .05, "ch_reserve_floor": .05,
    })


def test_contract_has_fresh_disjoint_calibration_and_gate_cohorts():
    contract = load_contract(CONTRACT)
    assert contract["calibration_seeds"] == list(range(12100, 12110))
    assert contract["reserved_gate_seeds"] == list(range(12200, 12220))
    assert not set(contract["calibration_seeds"]) & set(contract["reserved_gate_seeds"])
    assert len(expand_candidates(contract)) == 24


def test_teacher_scores_are_finite_and_action_is_exactly_feasible():
    env = Env(); mask = np.asarray([False, True, True, True, False]); caps = np.asarray([0, 2, 2, 1, 0])
    candidates = marginal_candidates(env, mask, caps, np.zeros(5, dtype=np.int64), config())
    assert candidates and all(np.isfinite(row["score"]) for row in candidates)
    action = marginal_utility_action(env, mask, caps, 4, config())
    assert int(action.sum()) == 4
    assert np.all(action <= caps) and np.all(action[~mask] == 0)


def test_dead_or_masked_nodes_never_receive_service():
    env = Env(); mask = np.asarray([False, False, True, False, False]); caps = np.asarray([0, 0, 2, 0, 0])
    action = marginal_utility_action(env, mask, caps, 24, config())
    assert action.tolist() == [0, 0, 2, 0, 0]


def test_teacher_is_equivariant_under_random_member_permutation():
    env = Env(); mask = np.asarray([False, True, True, True, True]); caps = np.asarray([0, 2, 2, 1, 1])
    reference = marginal_utility_action(env, mask, caps, 4, config())
    permutation = np.asarray([0, 3, 1, 4, 2])
    inverse = np.argsort(permutation)
    moved = Env()
    for name in ("positions", "energy", "queue"):
        setattr(moved.base, name, getattr(moved.base, name)[permutation])
    moved.base.packet_ages = [moved.base.packet_ages[index] for index in permutation]
    moved.step3_episode_offered_per_node = moved.step3_episode_offered_per_node[permutation]
    moved.step3_episode_delivered_per_node = moved.step3_episode_delivered_per_node[permutation]
    moved.ch = int(np.flatnonzero(permutation == env.ch)[0])
    permuted = marginal_utility_action(moved, mask[permutation], caps[permutation], 4, config())
    assert np.array_equal(permuted[inverse], reference)
