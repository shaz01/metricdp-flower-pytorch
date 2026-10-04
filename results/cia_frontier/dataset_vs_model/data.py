"""Stage A data modules: CIFAR-10 and EuroSAT with identical pipelines.

Both are RGB 32x32, no training-time augmentation, homogeneous partitions.
EuroSAT is downsized from 64x64 (bilinear) so the 32x32-only ``cifar10_cnn``
can run on it, and its usual random crop + flip is switched off because the
CIFAR-10 pipeline never had augmentation.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from experiments.reproduce.dataset.cifar10 import Cifar10DataModule
from experiments.reproduce.dataset.eurosat import EurosatDataModule

IMAGE_SIZE = (32, 32)


def _cache_dir(config: Mapping[str, Any]) -> str | None:
    return str(config.get("data-cache-dir", "")).strip() or None


def create_cifar10(config: Mapping[str, Any]) -> Cifar10DataModule:
    """Full 10-class CIFAR-10, native 32x32, no augmentation (unchanged plugin)."""
    return Cifar10DataModule(cache_dir=_cache_dir(config))


def create_eurosat32(config: Mapping[str, Any]) -> EurosatDataModule:
    """EuroSAT downsized to 32x32 with augmentation off, to match CIFAR-10."""
    return EurosatDataModule(cache_dir=_cache_dir(config), augment=False, resize_to=IMAGE_SIZE)


DATASETS = {
    "cifar10": f"{__name__}:create_cifar10",
    "eurosat32": f"{__name__}:create_eurosat32",
}
