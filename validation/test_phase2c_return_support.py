import math

import pytest

from experiments.audit_phase2c_return_support import (
    choose_scale,
    discounted_returns,
    quantile,
    validate_manifest,
)


def test_discounted_returns_matches_hand_trace():
    values = discounted_returns([1.0, 2.0, 3.0], gamma=0.5)
    assert values == pytest.approx([2.75, 3.5, 3.0])


def test_discounted_returns_terminal_singleton():
    assert discounted_returns([-2.0], gamma=0.99) == [-2.0]


def test_quantile_uses_linear_interpolation():
    assert quantile([0.0, 10.0], 0.25) == pytest.approx(2.5)


def test_scale_places_q_star_at_headroom():
    scale = choose_scale(q_star=80.0, v_max=30.0, headroom=0.8)
    assert scale == pytest.approx(0.3)
    assert scale * 80.0 == pytest.approx(24.0)


def test_scale_never_amplifies_rewards():
    assert choose_scale(q_star=10.0, v_max=30.0, headroom=0.8) == 1.0


def test_scale_rejects_nonpositive_tail():
    with pytest.raises(ValueError):
        choose_scale(q_star=0.0, v_max=30.0, headroom=0.8)
    with pytest.raises(ValueError):
        choose_scale(q_star=math.nan, v_max=30.0, headroom=0.8)


def test_manifest_rejects_held_out_overlap():
    manifest = {
        "development_seeds": [2300, 2301, 2302, 2303, 3100],
        "held_out_seeds_forbidden": [3100, 3101, 3102, 3103, 3104],
        "checkpoints": [{}, {}, {}],
        "max_steps": 300,
        "rollout_epsilon": 0.10,
    }
    with pytest.raises(ValueError, match="held-out seeds"):
        validate_manifest(manifest)