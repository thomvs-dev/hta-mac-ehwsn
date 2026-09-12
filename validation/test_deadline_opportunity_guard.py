import numpy as np
from agents.deadline_opportunity_guard import MotionHistory, cluster_action
from agents.feasible_service_dp import cluster_action as incumbent
from agents.temporal_reserve_dp import ReserveMemory
from envs.reviewer_energy_feasibility_fast import CostTable
from validation.test_reviewer_energy_feasibility import fixture, scalar_reference


def test_history_uses_copies_and_only_past_positions():
    history = MotionHistory()
    p = np.array([[1., 2.], [4., 6.]])
    np.testing.assert_array_equal(history.velocity(p), np.zeros_like(p))
    history.observe(p)
    p += 1
    np.testing.assert_array_equal(history.velocity(p), np.ones_like(p))


def test_expiring_packets_force_full_service_and_future_is_irrelevant():
    rng = np.random.default_rng(905100)
    for _ in range(80):
        base = fixture(rng.uniform(.01, 1.2, 3))
        base.cfg.n_max = 3
        base.cfg.packet_ttl_rounds = 3
        base.packet_ages = [[3]*3 for _ in range(3)]
        memory = ReserveMemory(3)
        table = CostTable(base)
        motion = MotionHistory()
        action = cluster_action(base, 0, memory, motion, table)
        np.testing.assert_array_equal(action, incumbent(base, 0, table))
        assert np.all(scalar_reference(base, action) <= base.energy + 1e-14)
        base.future_harvest = rng.uniform(0, 1000, 3)
        base.frozen_schedule = [{'positions': rng.uniform(-1000, 1000, (3, 2))}]
        np.testing.assert_array_equal(action, cluster_action(base, 0, memory, motion, table))


def test_nondeadline_actions_stay_on_existing_feasible_frontier():
    from agents.bounded_reserve_frontier import cluster_action as reserve
    rng = np.random.default_rng(905101)
    for _ in range(80):
        base = fixture(rng.uniform(.01, 1.2, 3))
        base.cfg.n_max = 3
        base.cfg.packet_ttl_rounds = 3
        base.packet_ages = [[2, 1, 0] for _ in range(3)]
        memory = ReserveMemory(3)
        motion = MotionHistory()
        motion.observe(base.positions + rng.normal(0, 1, base.positions.shape))
        table = CostTable(base)
        result = cluster_action(base, 0, memory, motion, table)
        assert any(np.array_equal(result, x) for x in [incumbent(base, 0, table), reserve(base, 0, memory, table, 'reserve_only')])
        assert np.all(scalar_reference(base, result) <= base.energy + 1e-14)
