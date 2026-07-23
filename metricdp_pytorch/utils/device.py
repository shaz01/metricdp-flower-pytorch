"""Shared accelerator selection for training and evaluation."""

from __future__ import annotations

import torch


def resolve_device() -> torch.device:
    """Return CUDA if available, else Apple MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
