from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agents.guarded_marginal_teacher import (
    GuardedTeacherConfig,
    guarded_marginal_action,
)
from experiments.run_v2_phase_b2_guarded_teacher_calibration import expand_candidates, load_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "v2_phase_b2_guarded_teacher_calibration_20260901.json"


class Radio:
    def tx(self, bits, distance): return bits * (1e-7 + distance * 1e-10)
    def rx(self, bits): return bits * 5e-8
    def aggregate(self, bits): return bits * 5e-9


class Env:
    def __init__(self):
        self.base = SimpleNamespace(
            n_nodes=5,
            cfg=SimpleNamespace(initial_energy_j=1.0, packet_bits=1000, packet_ttl_rounds=3, queue_max_packets=5, bs_position_m=(10., 10.)),
            positions=np.asarray([[0., 0.], [1., 0.], [2., 0.], [3., 0.], [4., 0.]]),
            energy=np.asarray([1., .9, .8, .7, .6]),
            packet_ages=[[0], [3, 1], [1, 0], [3], [0]],
            queue=np.asarray([1, 2, 2, 1, 1]), radio=Radio(),
        )
        self.base._state = lambda: np.column_stack((np.zeros(5), np.linspace(.1, .5, 5), np.zeros((5, 16))))
        self.ch = 0; self.members = np.asarray([1, 2, 3, 4])
        self.step3_episode_offered_per_node = np.asarray([0, 4, 4, 4, 4])
        self.step3_episode_delivered_per_node = np.asarray([0, 1, 3, 2, 4])
        self.step3_qos_counts = {"demand": 100, "stale": 3}


def config():
    return GuardedTeacherConfig.from_payload({
        "energy_weight": 2, "depletion_weight": 1.5, "fairness_guard": .92,
        "stale_guard": .03, "reserve_floor": .08,
    })


def test_b2_contract_has_new_disjoint_cohorts_and_exact_gate_targets():
    contract = load_contract(CONTRACT)
    assert contract["calibration_seeds"] == list(range(12100, 12110))
    assert contract["reserved_gate_seeds"] == list(range(12400, 12420))
    assert set(range(3900, 3920)) <= set(contract["opened_or_prior_seeds"])
    assert set(range(12200, 12220)) <= set(contract["opened_or_prior_seeds"])
    assert len(expand_candidates(contract)) == 18
    assert contract["gate_b_targets"] == {
        "delivery_min": .45, "packets_per_j_min": 244.8, "stale_ratio_max": .035,
        "fairness_min": .92, "rmst_min": 128., "feasibility_violations_max": 0,
    }


def test_guarded_teacher_is_finite_exact_and_forces_expiring_service_at_guard():
    env = Env(); mask = np.asarray([False, True, True, True, True]); caps = np.asarray([0, 2, 2, 1, 1])
    action, audit = guarded_marginal_action(env, mask, caps, 2, config(), return_audit=True)
    assert int(action.sum()) == 2 and np.all(action <= caps) and np.all(action[~mask] == 0)
    assert audit["stale_guard_satisfied"] and audit["served_expiring"] >= audit["required_expiring_service"]


def test_guarded_teacher_is_equivariant_under_member_permutation():
    env = Env(); mask = np.asarray([False, True, True, True, True]); caps = np.asarray([0, 2, 2, 1, 1])
    reference = guarded_marginal_action(env, mask, caps, 4, config())
    permutation = np.asarray([0, 3, 1, 4, 2]); inverse = np.argsort(permutation); moved = Env()
    for name in ("positions", "energy", "queue"):
        setattr(moved.base, name, getattr(moved.base, name)[permutation])
    moved.base.packet_ages = [moved.base.packet_ages[index] for index in permutation]
    moved.step3_episode_offered_per_node = moved.step3_episode_offered_per_node[permutation]
    moved.step3_episode_delivered_per_node = moved.step3_episode_delivered_per_node[permutation]
    moved.members = np.asarray([int(np.flatnonzero(permutation == node)[0]) for node in env.members])
    moved.ch = int(np.flatnonzero(permutation == env.ch)[0])
    permuted = guarded_marginal_action(moved, mask[permutation], caps[permutation], 4, config())
    assert np.array_equal(permuted[inverse], reference)


def test_guarded_teacher_handles_terminal_empty_mask_without_normalization():
    env = Env(); mask = np.zeros(5, dtype=bool); caps = np.zeros(5, dtype=np.int64)
    action, audit = guarded_marginal_action(env, mask, caps, 4, config(), return_audit=True)
    assert np.array_equal(action, np.zeros(5, dtype=np.int64))
    assert audit["target_slots"] == 0 and audit["stale_guard_satisfied"]
