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


def release_device_cache(device: torch.device) -> None:
    """Return pooled accelerator memory to the OS after a train/eval task.

    CUDA's and MPS's caching allocators reuse freed tensor memory instead of
    releasing it back to the OS. Flower's Ray simulation backend keeps each
    client as a long-lived actor process for the whole run, so on MPS (unified
    memory, shared with system RAM) that pooled-but-idle memory accumulates
    round over round within one actor until it exceeds total system memory on
    long/high-round-count runs -- call this once per task after the model is
    done with the device to keep each actor's footprint bounded.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
