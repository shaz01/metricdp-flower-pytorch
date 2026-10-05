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
EUROSAT_TRAIN_SIZE = 21_600  # EuroSAT's training split; CIFAR-10 is cut to this for Stage B


def _cache_dir(config: Mapping[str, Any]) -> str | None:
    return str(config.get("data-cache-dir", "")).strip() or None


def create_cifar10(config: Mapping[str, Any]) -> Cifar10DataModule:
    """Full 10-class CIFAR-10, native 32x32, no augmentation (unchanged plugin)."""
    return Cifar10DataModule(cache_dir=_cache_dir(config))


def create_eurosat32(config: Mapping[str, Any]) -> EurosatDataModule:
    """EuroSAT downsized to 32x32 with augmentation off, to match CIFAR-10."""
    return EurosatDataModule(cache_dir=_cache_dir(config), augment=False, resize_to=IMAGE_SIZE)


def create_cifar10_small(config: Mapping[str, Any]) -> Cifar10DataModule:
    """CIFAR-10 cut to a fixed stratified 21,600-image pool before partitioning.

    Matches EuroSAT's pool size so both datasets give clients the same amount of data
    under any partition, including Dirichlet. Pool is fixed across training seeds.
    """
    return Cifar10DataModule(cache_dir=_cache_dir(config), train_subsample=EUROSAT_TRAIN_SIZE,
                             subsample_seed=0)


DATASETS = {
    "cifar10": f"{__name__}:create_cifar10",
    "eurosat32": f"{__name__}:create_eurosat32",
}
STAGE_B_DATASETS = {
    "cifar10s": f"{__name__}:create_cifar10_small",
    "eurosat32": f"{__name__}:create_eurosat32",
}
