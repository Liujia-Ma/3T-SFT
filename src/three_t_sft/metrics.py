"""Numerical summaries for independently repeated experiments."""
from math import isfinite
from statistics import mean, stdev


def summarize_seeds(values):
    values = list(values)
    if not values or any(not isfinite(v) for v in values):
        raise ValueError('Finite nonempty measurements required')
    return {'count': len(values), 'mean': mean(values),
            'sample_std': stdev(values) if len(values)>1 else None}


def paired_gains(treatment, baseline):
    if set(treatment) != set(baseline): raise ValueError('Seed IDs must match')
    return summarize_seeds(treatment[seed]-baseline[seed] for seed in sorted(treatment))
