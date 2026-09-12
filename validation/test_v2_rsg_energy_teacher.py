from pathlib import Path
from types import SimpleNamespace

import numpy as np

from agents.rsg_energy_teacher import RSGEnergyConfig, RSGEnergyTeacher
from experiments.run_v2_phase_b3_literature_rsg_energy_calibration import expand_candidates, load_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "v2_phase_b3_literature_rsg_energy_calibration_20260904.json"


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
            energy=np.asarray([1., .9, .8, .7, .6]), alive=np.ones(5, dtype=bool),
            packet_ages=[[0], [3, 1], [1, 0], [3], [0]],
            queue=np.asarray([1, 2, 2, 1, 1]), radio=Radio(),
        )
        self.base._state = lambda: np.column_stack((np.zeros(5), np.linspace(.1, .5, 5), np.zeros((5, 16))))
        self.ch = 0; self.members = np.asarray([1, 2, 3, 4])
        self.step3_episode_offered_per_node = np.asarray([0, 4, 4, 4, 4])
        self.step3_episode_delivered_per_node = np.asarray([0, 1, 3, 2, 4])
        self.step3_qos_counts = {"demand": 100, "stale": 3}


def config():
    return RSGEnergyConfig.from_payload({
        "energy_weight": 2, "deadline_weight": 4, "tsls_weight": 1.5,
        "harvest_credit": 1, "queue_weight": 1, "delivery_debt_weight": 1,
        "depletion_weight": 1, "stale_guard": .035, "fairness_guard": .92,
        "reserve_floor": .08, "delivery_floor": .45,
    })


def test_b3_contract_keeps_gate_targets_and_seed_boundaries():
    contract = load_contract(CONTRACT)
    assert contract["development_seeds"] == list(range(12100, 12110))
    assert contract["reserved_gate_seeds"] == list(range(12500, 12520))
    assert not set(range(3900, 3920)) & set(contract["development_seeds"])
    assert len(expand_candidates(contract)) == 24
    assert contract["gate_b_targets"] == {
        "delivery_min": .45, "packets_per_j_min": 244.8, "stale_ratio_max": .035,
        "fairness_min": .92, "rmst_min": 128., "feasibility_violations_max": 0,
    }


def test_rsg_teacher_is_exact_stateful_and_guarded():
    env = Env(); mask = np.asarray([False, True, True, True, True]); caps = np.asarray([0, 2, 2, 1, 1])
    teacher = RSGEnergyTeacher(5, config())
    action, audit = teacher.action(env, mask, caps, 2, return_audit=True)
    assert int(action.sum()) == 2 and np.all(action <= caps) and np.all(action[~mask] == 0)
    assert audit["stale_guard_satisfied"] and teacher.round_index == 1
    assert np.all(teacher.tsls[action > 0] == 0)


def test_rsg_teacher_is_permutation_equivariant():
    env = Env(); mask = np.asarray([False, True, True, True, True]); caps = np.asarray([0, 2, 2, 1, 1])
    first = RSGEnergyTeacher(5, config()); first.tsls = np.asarray([0, 4, 1, 3, 2])
    reference = first.action(env, mask, caps, 4)
    permutation = np.asarray([0, 3, 1, 4, 2]); inverse = np.argsort(permutation); moved = Env()
    for name in ("positions", "energy", "alive", "queue"):
        setattr(moved.base, name, getattr(moved.base, name)[permutation])
    moved.base.packet_ages = [moved.base.packet_ages[index] for index in permutation]
    moved.step3_episode_offered_per_node = moved.step3_episode_offered_per_node[permutation]
    moved.step3_episode_delivered_per_node = moved.step3_episode_delivered_per_node[permutation]
    moved.members = np.asarray([int(np.flatnonzero(permutation == node)[0]) for node in env.members])
    moved.ch = int(np.flatnonzero(permutation == env.ch)[0])
    second = RSGEnergyTeacher(5, config()); second.tsls = np.asarray([0, 4, 1, 3, 2])[permutation]
    permuted = second.action(moved, mask[permutation], caps[permutation], 4)
    assert np.array_equal(permuted[inverse], reference)
