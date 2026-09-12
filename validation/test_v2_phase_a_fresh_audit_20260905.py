import numpy as np
import pytest
from experiments.run_v2_phase_a_fresh_audit_20260905 import controlled_energy, validate_action, gate_a_attribution


def test_shared_forwarding_and_idle_activation_are_not_linear_per_packet():
    # Two packets on one member incur one setup and no idle. Activating a
    # second member adds idle on both active members for the three-slot frame.
    cost = np.array([2.0, 3.0])
    one = controlled_energy(np.array([2, 0]), cost, 7.0, 0.5)
    two = controlled_energy(np.array([2, 1]), cost, 7.0, 0.5)
    assert one == 11.0
    assert two == 15.5
    assert two - one == 4.5
    assert controlled_energy(np.zeros(2), cost, 7.0, 0.5) == 0.0


@pytest.mark.parametrize("action", [[2, 1], [0, 1], [-1, 0], [1.5, 0], [float('nan'), 0], [4, 0]])
def test_invalid_actions_fail_before_environment_masking(action):
    with pytest.raises(RuntimeError):
        validate_action(np.array(action), np.array([True, False]), np.array([3, 0]), 2)


def test_valid_underfull_action_is_allowed():
    validate_action(np.array([1, 0]), np.array([True, False]), np.array([3, 0]), 2)


def test_gate_formula_remains_identical_to_original():
    from experiments.run_v2_phase_a_mechanism_audit import gate_a_attribution as original
    def row(d, p, e, u):
        return dict(mean_delivery_ratio=d, mean_global_packets=p, mean_network_energy_j=e,
                    mean_under_service_capacity_ratio=u, max_role_reconstruction_error_j=0.)
    values = dict(hta_mac_v1=row(.4, 100, 1, .01), residual_energy_cap_corrected=row(.45, 120, 1.1, 0))
    assert gate_a_attribution(values, .8) == original(values, .8)
    assert not gate_a_attribution(values, .8)["gate_a_pass"]
