import pytest
from envs.censored_packet_bounds import outcome_bounds


def test_every_possible_pending_resolution_lies_inside_bounds():
    for delivered in range(4):
        for pending in range(5):
            c=dict(generated=delivered+pending+3,delivered=delivered,pending=pending,stale=1,death=1,overflow=1)
            b=outcome_bounds(c)
            for success in range(pending+1):
                assert b['delivery_lower'] <= (delivered+success)/c['generated'] <= b['delivery_upper']
                assert b['stale_lower'] <= (1+success)/c['generated'] <= b['stale_upper']


def test_missing_births_are_undefined_and_corrupt_counts_rejected():
    empty=dict.fromkeys(['generated','delivered','stale','death','overflow','pending'],0)
    assert not outcome_bounds(empty)['defined']
    assert outcome_bounds(empty)['delivery_lower'] is None
    with pytest.raises(ValueError):
        outcome_bounds(dict(empty,generated=1))
