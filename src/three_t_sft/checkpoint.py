"""Atomic checkpoint replacement avoids leaving a truncated selected model."""
import os
import tempfile
from pathlib import Path
import torch


def atomic_save(state, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix='.checkpoint-', suffix='.tmp', dir=destination.parent)
    os.close(handle)
    try:
        torch.save(state, name)
        os.replace(name, destination)
    finally:
        if os.path.exists(name): os.unlink(name)
