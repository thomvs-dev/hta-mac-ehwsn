import numpy as np
import pytest
from core.energy.radio_model import RadioModel
from envs.control_cost_envelope import full_reporting_cost


def test_independent_scalar_roles_and_ack_bounds():
    r=RadioModel(.01,.001,.0001,.002,10.)
    # Two members at 1m and 2m from the same CH. Ten-bit messages.
    x=full_reporting_cost([[1,0],[2,0],[0,0]],[2],r,10)
    np.testing.assert_allclose(x['exchange_j'],[.21,.24,.45])
    np.testing.assert_allclose(x['listen_j'],[.2,.2,0])
    assert x['ack_min_j']==pytest.approx(.2)
    assert x['ack_max_j']==pytest.approx(.24)
    assert sum(v.sum() for v in x['role_j'].values())==pytest.approx(1.3)


def test_permutation_and_message_scaling():
    r=RadioModel(.01,.001,.0001,.002,10.)
    pos=np.array([[1,0],[2,0],[0,0],[20,0]],float)
    a=full_reporting_cost(pos,[2,3],r,10);p=np.array([2,0,3,1]);inverse=np.argsort(p)
    b=full_reporting_cost(pos[p],inverse[[2,3]],r,20)
    np.testing.assert_allclose(b['exchange_j'][inverse],2*a['exchange_j'])
    np.testing.assert_allclose(b['listen_j'][inverse],2*a['listen_j'])
    assert a['members']==b['members']==2


def test_invalid_geometry_and_sizes():
    r=RadioModel(.01,.001,.0001,.002,10.)
    for pos,heads,bits in [([[float('nan'),0]],[0],10),([[0,0]],[],10),([[0,0]],[0,0],10),([[0,0]],[0],0)]:
        with pytest.raises(ValueError):full_reporting_cost(pos,heads,r,bits)
