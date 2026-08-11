import numpy as np

from experiments.diagnose_step3_delivery_feasibility import (
    fair_budget_fill,
    unconstrained_argmax,
)
from experiments.sweep_step3_risk_gated_completion import risk_gated_complete
from experiments.sweep_step3_qos_deficit_override import qos_deficit_override
from experiments.sweep_step3_qos_band_projection import qos_band_projection


def test_fair_budget_fill_uses_budget_without_exceeding_caps():
    caps = np.array([3, 2, 0, 4])
    mask = np.array([True, True, False, True])
    action = fair_budget_fill(caps, mask, 6, np.zeros(4))
    assert action.sum() == 6
    assert np.all(action <= caps)
    assert action[2] == 0


def test_fair_budget_fill_stops_when_backlog_is_exhausted():
    action = fair_budget_fill(
        np.array([1, 0, 2]), np.array([True, True, True]), 16, np.zeros(3)
    )
    assert action.tolist() == [1, 0, 2]


def test_unconstrained_argmax_respects_mask_and_caps():
    q = np.array([[0.0, 3.0, 9.0], [0.0, 8.0, 1.0], [4.0, 2.0, 1.0]])
    action = unconstrained_argmax(q, np.array([1, 2, 2]), np.array([True, True, False]))
    assert action.tolist() == [1, 1, 0]


class _Radio:
    def rx(self, bits): return bits * 1e-6
    def aggregate(self, bits): return bits * 1e-6
    def tx(self, bits, distance): return bits * 1e-6


class _Cfg:
    packet_bits = 1
    frame_slot_budget = 4
    initial_energy_j = 1.0
    bs_position_m = (0.0, 0.0)


class _Base:
    cfg = _Cfg()
    radio = _Radio()
    positions = np.zeros((2, 2))
    alive = np.ones(2, dtype=bool)
    energy = np.ones(2)


class _Env:
    base = _Base()
    ch = 0


def test_risk_gated_completion_adds_only_frozen_fraction():
    q = np.array([[0.0, -0.001, -0.002], [0.0, -0.001, -0.002]])
    action, audit = risk_gated_complete(
        np.zeros(2, dtype=np.int64), q, np.array([2, 2]),
        np.array([True, True]), _Env(), reserve_floor=0.2,
        completion_fraction=0.5, negative_tolerance_factor=2.0,
    )
    assert action.sum() == 2
    assert audit["added"] == 2


class _QoSEnv(_Env):
    members = np.array([0, 1])
    step3_qos_counts = {"delivered": 2, "demand": 10}
    _Base.queue = np.array([2, 2])


def test_qos_deficit_override_ignores_negative_q_but_respects_deficit():
    q = np.array([[0.0, -10.0, -20.0], [0.0, -9.0, -18.0]])
    action, audit = qos_deficit_override(
        np.zeros(2, dtype=np.int64), q, np.array([2, 2]),
        np.array([True, True]), _QoSEnv(), trajectory_target=0.5,
        reserve_floor=0.2, completion_fraction=1.0,
    )
    assert audit["deficit_after_base"] == 5
    assert action.sum() == 4
    assert audit["added"] == 4


class _BandEnv(_QoSEnv):
    step3_qos_counts = {"delivered": 5, "demand": 10}
    cumulative_service = np.array([10.0, 1.0])


def test_qos_band_projection_removes_only_upper_excess_from_overserved_node():
    q = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    action, audit = qos_band_projection(
        np.array([2, 2]), q, np.array([2, 2]), np.array([True, True]),
        _BandEnv(), lower_target=0.5, upper_target=0.6,
        reserve_floor=0.2, completion_fraction=1.0,
    )
    assert action.tolist() == [1, 2]
    assert audit["removed"] == 1
    assert audit["added"] == 0
