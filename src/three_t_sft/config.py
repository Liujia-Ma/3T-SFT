"""Validate experiment settings before files or model parameters are created."""
from math import isfinite


def validate_config(config):
    for key in ('steps', 'eval_every', 'max_tokens'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if type(config.get('seed')) is not int: raise ValueError('seed must be an integer')
    if type(config.get('heterogeneous')) is not bool: raise ValueError('heterogeneous must be boolean')
    if config.get('mode') not in ('radst', 'terminal'): raise ValueError('Unknown training mode')
    for key in ('learning_rate', 'temperature', 'grad_clip'):
        value = config.get(key)
        if type(value) not in (int, float) or not isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
    return dict(config)
