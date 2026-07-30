"""Unit tests for the shared Phase 3 MAC policy interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from baselines import (
    EnergyProportionalPolicy,
    FFSSAdaptedPolicy,
    HTAMACPolicy,
    HarvestProportionalPolicy,
    RandomBudgetedPolicy,
    S2A2MACAdaptedPolicy,
    StaticEqualPolicy,
    phase3_policy_factories,
)
from core.energy.radio_model import RadioModel
from experiments.run_phase3_pilot import (
    censor_aware_lifetime_summary,
    kaplan_meier_median,
)


class FakeConfig:
    initial_energy_j = 0.5
    queue_max_packets = 5
    frame_slot_budget = 5
    n_max = 3
    packet_bits = 4000


class FakeEnv:
    def __init__(self):
        self.cfg = FakeConfig()
        self.n_nodes = 8
        self.cluster_heads = np.array([0, 4])
        self.cluster_of = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        self.alive = np.ones(8, dtype=bool)
        self.energy = np.array([0.5, 0.1, 0.3, 0.5, 0.5, 0.2, 0.4, 0.5])
        self.queue = np.array([1, 1, 2, 4, 1, 2, 3, 5])
        self.positions = np.column_stack((np.arange(8), np.zeros(8)))
        self.embedding = np.zeros((8, 32), dtype=np.float32)
        self.round = 0
        self.radio = RadioModel(
            e_elec_j_per_bit=5e-8,
            eps_fs_j_per_bit_m2=1e-11,
            eps_mp_j_per_bit_m4=1.3e-15,
            e_da_j_per_bit=5e-9,
            d0_m=87.70580193070292,
        )

    def static_equal_action(self):
        action = np.zeros(self.n_nodes, dtype=np.int64)
        for cluster, ch in enumerate(self.cluster_heads):
            if not self.alive[ch]:
                continue
            members = np.flatnonzero(
                (self.cluster_of == cluster)
                & self.alive
                & (np.arange(self.n_nodes) != ch)
            )
            action[members[: self.cfg.frame_slot_budget]] = 1
        return action

    def _validate_action(self, action):
        assert action.shape == (self.n_nodes,)
        assert np.all((action >= 0) & (action <= self.cfg.n_max))
        assert np.all(action[self.cluster_heads] == 0)
        assert np.all(action[~self.alive] == 0)
        for cluster, ch in enumerate(self.cluster_heads):
            members = self.cluster_of == cluster
            assert int(action[members].sum()) <= self.cfg.frame_slot_budget
            if not self.alive[ch]:
                assert np.all(action[members] == 0)


def state():
    value = np.zeros((8, 18), dtype=np.float32)
    value[:, 0] = 1.0
    value[:, 1] = np.linspace(0.01, 0.08, 8)
    value[:, 15] = 0.5
    return value


def test_all_seven_policies_are_identified_once():
    root = Path(__file__).resolve().parents[1]
    names = [name for name, _ in phase3_policy_factories(root)]
    assert len(names) == 7
    assert len(set(names)) == 7
    assert names[-1] == "random_budgeted_diagnostic"
    assert RandomBudgetedPolicy.comparison_role == "formal_stochastic_floor"


def test_heuristic_policies_share_action_contract():
    env = FakeEnv()
    policies = (
        StaticEqualPolicy(),
        EnergyProportionalPolicy(),
        HarvestProportionalPolicy(),
        S2A2MACAdaptedPolicy(),
        FFSSAdaptedPolicy(),
        RandomBudgetedPolicy(),
    )
    for policy in policies:
        policy.reset(77)
        action = policy.select_action(state(), env)
        env._validate_action(action)


def test_dead_cluster_head_suppresses_member_allocations():
    env = FakeEnv()
    env.alive[4] = False
    for policy in (
        StaticEqualPolicy(),
        EnergyProportionalPolicy(),
        HarvestProportionalPolicy(),
        S2A2MACAdaptedPolicy(),
        FFSSAdaptedPolicy(),
        RandomBudgetedPolicy(),
    ):
        action = policy.select_action(state(), env)
        assert np.all(action[env.cluster_of == 1] == 0)


def test_s2a2mac_alternates_clusters_and_uses_three_levels():
    env = FakeEnv()
    policy = S2A2MACAdaptedPolicy()
    action_round_zero = policy.select_action(state(), env)
    assert np.all(action_round_zero[env.cluster_of == 0] == 0)
    assert np.any(action_round_zero[env.cluster_of == 1] > 0)
    env.round = 1
    action_round_one = policy.select_action(state(), env)
    assert np.any(action_round_one[env.cluster_of == 0] > 0)
    assert np.all(action_round_one[env.cluster_of == 1] == 0)
    assert int(action_round_one.max()) >= 2


def test_ffss_adaptation_assigns_at_most_one_slot_per_node():
    action = FFSSAdaptedPolicy().select_action(state(), FakeEnv())
    assert np.all(action <= 1)


def test_frozen_hta_checkpoint_obeys_shared_contract():
    root = Path(__file__).resolve().parents[1]
    checkpoint = (
        root
        / "outputs"
        / "phase2"
        / "authoritative_dynamic_budget8_500ep"
        / "branching_c51.pt"
    )
    policy = HTAMACPolicy(checkpoint)
    env = FakeEnv()
    action = policy.select_action(state(), env)
    env._validate_action(action)
    members = ~np.isin(np.arange(env.n_nodes), env.cluster_heads)
    assert np.all(action[members] <= env.queue[members])


def test_kaplan_meier_median_is_unreached_when_all_trials_censored():
    assert kaplan_meier_median([10, 12], [False, False]) is None
    assert kaplan_meier_median([3, 4], [True, True]) == 3.0


def test_censor_aware_summary_uses_common_restricted_horizon():
    rows = []
    for seed, censor_round in ((1, 10), (2, 12)):
        rows.append(
            {
                "seed": seed,
                "policy": "hta_mac",
                "t_fnd": None,
                "t_hnd": None,
                "t_fnd_event_observed": False,
                "t_hnd_event_observed": False,
                "censor_round": censor_round,
            }
        )
    for seed, event_time, censor_round in ((1, 3, 10), (2, 4, 12)):
        rows.append(
            {
                "seed": seed,
                "policy": "static_equal",
                "t_fnd": event_time,
                "t_hnd": event_time + 1,
                "t_fnd_event_observed": True,
                "t_hnd_event_observed": True,
                "censor_round": censor_round,
            }
        )
    summary = censor_aware_lifetime_summary(rows)
    assert summary["common_restriction_round"] == 10
    hta = summary["endpoints"]["t_fnd"]["hta_mac"]
    static = summary["endpoints"]["t_fnd"]["static_equal"]
    assert hta["events"] == 0
    assert hta["right_censored"] == 2
    assert hta["kaplan_meier_median_round"] is None
    assert hta["restricted_mean_event_free_rounds"] == 10.0
    assert static["restricted_mean_event_free_rounds"] == 3.5
    assert (
        static["paired_wilcoxon_vs_hta"][
            "median_paired_difference_rounds"
        ]
        == 6.5
    )
