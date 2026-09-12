import numpy as np
from core.energy.idle_model import energy_update
from experiments.audit_reviewer_foundation_20260906 import deficit_audit


def test_clipping_order_is_material():
    actual=energy_update([.1],[.2],[.15],1.)
    np.testing.assert_allclose(actual,[.05])
    assert abs(actual[0]-.15)>.09


def test_receiver_deficit_does_not_double_count_packet():
    a=deficit_audit(np.array([.1,.1]),np.array([.3,.3]),np.zeros(2),np.array([2,0]),np.array([0,0]),np.array([1]))
    assert a['even_with_harvest_deficit_nodes']==2
    assert a['even_with_harvest_associated_packets']==2


def test_harvest_timing_distinction():
    a=deficit_audit(np.array([.1,.9]),np.array([.2,.1]),np.array([.2,0]),np.array([1,0]),np.array([0,0]),np.array([1]))
    assert a['pre_harvest_associated_packets']==1
    assert a['even_with_harvest_associated_packets']==0
