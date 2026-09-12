from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.run_v2_phase_b5_reduced_cluster_exact_oracle import exact_reduced_oracle, load_contract


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "config" / "v2_phase_b5_reduced_cluster_exact_oracle_20260904.json"


class Radio:
    def tx(self, bits, distance): return bits * (1e-7 + distance * 1e-10)
    def rx(self, bits): return bits * 5e-8
    def aggregate(self, bits): return bits * 5e-9


class Env:
    def __init__(self):
        self.base = SimpleNamespace(
            n_nodes=5,
            cfg=SimpleNamespace(initial_energy_j=1.0, packet_bits=1000, packet_ttl_rounds=3, queue_max_packets=5, bs_position_m=(10., 10.), e_elec_j_per_bit=5e-8, idle_slot_bit_times=1000),
            positions=np.asarray([[0., 0.], [8., 0.], [1., 0.], [3., 0.], [4., 0.]]),
            energy=np.asarray([1., .9, .9, .9, .9]), alive=np.ones(5, dtype=bool),
            packet_ages=[[0], [1], [1, 0], [3], [1, 0, 0]],
            queue=np.asarray([1, 1, 2, 1, 3]), radio=Radio(),
        )
        self.ch = 0; self.members = np.asarray([1, 2, 3, 4])
        self.step3_episode_offered_per_node = np.asarray([0, 10, 10, 10, 10])
        self.step3_episode_delivered_per_node = np.asarray([0, 5, 5, 5, 5])
        self.step3_qos_counts = {"demand": 100, "stale": 2}


def test_b5_contract_is_no_training_and_excludes_confirmation_seeds():
    contract = load_contract(CONTRACT)
    assert contract["decision_rules"]["no_training"]
    assert not set(contract["development_seeds"]) & set(range(3900, 3920))
    assert contract["continuation_gate"]["authorizes"].startswith("new deterministic")


def test_exact_reduced_oracle_never_degrades_registered_constraints():
    env = Env(); mask = np.asarray([False, True, True, True, True]); caps = np.asarray([0, 1, 2, 1, 3])
    base = np.asarray([0, 1, 0, 1, 0])
    oracle, audit = exact_reduced_oracle(env, base, caps, mask, 4)
    assert int(oracle.sum()) == int(base.sum())
    assert audit["oracle_expiring"] >= audit["base_expiring"]
    assert audit["oracle_fairness"] + 1e-12 >= audit["base_fairness"]
    assert audit["oracle_minimum_post_energy"] + 1e-15 >= audit["base_minimum_post_energy"]
    assert audit["oracle_reduced_tx_idle_j"] <= audit["base_reduced_tx_idle_j"] + 1e-15
