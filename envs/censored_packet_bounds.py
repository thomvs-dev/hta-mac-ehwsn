"""Identification bounds for already-generated packets, not future arrivals."""
def outcome_bounds(cohort):
    keys = ['generated', 'delivered', 'stale', 'death', 'overflow', 'pending']
    if any(cohort[k] < 0 or int(cohort[k]) != cohort[k] for k in keys):
        raise ValueError('invalid packet counts')
    n = cohort['generated']
    if n != sum(cohort[k] for k in keys[1:]):
        raise ValueError('packet conservation failed')
    if n == 0:
        return dict(defined=False, delivery_lower=None, delivery_upper=None, stale_lower=None, stale_upper=None)
    return dict(defined=True, delivery_lower=cohort['delivered']/n,
                delivery_upper=(cohort['delivered']+cohort['pending'])/n,
                stale_lower=cohort['stale']/n,
                stale_upper=(cohort['stale']+cohort['pending'])/n)
