"""Replay stochastic node computation without advancing the caller RNG."""
from contextlib import contextmanager
from dataclasses import dataclass
import torch


@dataclass
class RNGState:
    cpu: torch.Tensor
    cuda: list[torch.Tensor]

    @classmethod
    def capture(cls):
        return cls(torch.get_rng_state().clone(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

    def restore(self):
        torch.set_rng_state(self.cpu)
        if self.cuda: torch.cuda.set_rng_state_all(self.cuda)


@contextmanager
def replay_rng(state):
    current = RNGState.capture()
    try:
        state.restore()
        yield
    finally:
        current.restore()
