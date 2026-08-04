"""Shared Hugging Face dataset loading helpers."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any, Literal

import torch
from datasets import DatasetDict, load_dataset
from PIL import Image
from torchvision.transforms import ToTensor

HF_OFFLINE_VARS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")
DatasetLoader = Callable[..., Any]
PartitionMode = Literal["homogeneous", "non-iid"]
_TO_TENSOR = ToTensor()


def grayscale_image_transform(
    image_size: tuple[int, int],
) -> Callable[[Any], torch.Tensor]:
    """Create a transform for validated grayscale image tensors."""
    def transform(image: Any) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("The image column must decode to a PIL image.")
        grayscale = image.convert("L")
        if grayscale.size != image_size:
            raise ValueError(
                f"Expected {image_size[0]}×{image_size[1]} images, "
                f"got {grayscale.size}."
            )
        return _TO_TENSOR(grayscale)

    return transform


@contextmanager
def hf_offline() -> Iterator[None]:
    """Temporarily force Hugging Face Hub and datasets lookups offline."""
    previous = {name: os.environ.get(name) for name in HF_OFFLINE_VARS}
    os.environ.update({name: "1" for name in HF_OFFLINE_VARS})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@cache
def _load_hf_dataset_once(
    dataset_id: str,
    cache_dir: str | None,
    loader: DatasetLoader,
) -> DatasetDict:
    """Materialize one shared dataset for an ID and resolved cache location."""
    if any(os.environ.get(name) is not None for name in HF_OFFLINE_VARS):
        return loader(dataset_id, cache_dir=cache_dir)
    try:
        with hf_offline():
            return loader(dataset_id, cache_dir=cache_dir)
    except Exception:  # noqa: BLE001 - any cache miss should retry over network
        return loader(dataset_id, cache_dir=cache_dir)


def load_hf_dataset_cached(
    dataset_id: str,
    cache_dir: str | Path | None = None,
    *,
    loader: DatasetLoader = load_dataset,
) -> DatasetDict:
    """Load a dataset cache-first and reuse it process-wide.

    Unless the caller explicitly configured Hugging Face offline mode, the
    first lookup is forced offline so an existing cache does not perform a Hub
    check. A cache miss is retried with normal network access. Explicit
    ``HF_*_OFFLINE`` settings are always respected.
    """
    resolved_cache_dir = None if cache_dir is None else str(cache_dir)
    return _load_hf_dataset_once(dataset_id, resolved_cache_dir, loader)


def clear_hf_dataset_cache() -> None:
    """Clear process-wide dataset singletons (primarily useful in tests)."""
    _load_hf_dataset_once.cache_clear()
