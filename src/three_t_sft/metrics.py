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


def parameter_counts(model):
    """PyTorch parameter iteration deduplicates shared parameter objects."""
    params = list(model.parameters())
    total = sum(p.numel() for p in params)
    trainable = sum(p.numel() for p in params if p.requires_grad)
    return {'total': total, 'trainable': trainable, 'frozen': total-trainable}


def gradient_norm(parameters):
    """Global L2 norm before clipping; fail on invalid optimization signals."""
    import torch
    norms = []
    for parameter in parameters:
        if parameter.grad is None: continue
        if not torch.isfinite(parameter.grad).all(): raise ValueError('Nonfinite parameter gradient')
        norms.append(parameter.grad.detach().double().norm().item())
    return sum(value*value for value in norms)**.5
