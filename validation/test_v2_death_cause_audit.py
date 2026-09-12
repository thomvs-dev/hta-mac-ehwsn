import numpy as np
from experiments.run_v2_death_cause_audit_20260905 import role_label,death_records


def test_role_partition():
    assert [role_label(n,[1,2],0,[0,3]) for n in range(5)]==[
        'target_ch','target_member','target_member','background_ch','background_member']


def test_new_deaths_only_and_controlled_accounting():
    roles=np.ones((5,4)); controlled=roles*.25
    rows=death_records(np.array([1,1,0,1],bool),np.array([0,1,0,0],bool),[1],0,[0,3],
                       np.ones(4),np.zeros(4),roles,controlled,roles,8,True)
    assert [r['node'] for r in rows]==[0,3]
    assert all(r['cumulative_controlled_fraction']==.25 for r in rows)
    assert rows[1]['role_at_death']=='background_ch'
