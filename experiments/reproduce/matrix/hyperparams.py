"""Shared reproduction training and privacy hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Hyperparams:
    """Training and privacy hyperparameters shared by reproduction runs."""

    noise_multiplier: float
    clipping_norm: float
    rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    initialization_epochs: int
