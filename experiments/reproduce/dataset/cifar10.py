"""Full ten-class CIFAR-10 data plugin for reproduction experiments.

The four-class variant used by the original reproduction runs lives in
``experiments.reproduce.dataset.cifar4``. Pair this plugin with
``experiments.reproduce.cifar10_cnn:create_model``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from experiments.reproduce.dataset.common import PartitionMode, load_hf_dataset_cached
from metricdp_pytorch.utils.data import (
    RecordImageDataset,
    labels_from_records,
    make_client_loaders,
    make_server_loaders,
)
from metricdp_pytorch.utils.split_data import (
    balanced_stratified_partitions,
    label_shard_partitions,
    quantity_skewed_partitions,
)

DATASET_ID = "uoft-cs/cifar10"
IMAGE_SIZE = (32, 32)
IMAGE_COLUMN = "img"
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CLASS_IDS = tuple(range(len(CLASS_NAMES)))
_TO_TENSOR = ToTensor()


def load_cifar10_dataset(cache_dir: str | Path | None = None) -> DatasetDict:
    """Return the unfiltered ten-class CIFAR-10 dataset."""
    return load_hf_dataset_cached(DATASET_ID, cache_dir)


@dataclass(frozen=True)
class RGBImageTransform:
    """Pickle-safe validated RGB tensor transform."""

    image_size: tuple[int, int]

    def __call__(self, image: Any) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("The image column must decode to a PIL image.")
        rgb = image.convert("RGB")
        if rgb.size != self.image_size:
            raise ValueError(
                f"Expected {self.image_size[0]}×{self.image_size[1]} images, "
                f"got {rgb.size}."
            )
        return _TO_TENSOR(rgb)


_CIFAR10_IMAGE_TRANSFORM = RGBImageTransform(IMAGE_SIZE)


class Cifar10Dataset(RecordImageDataset):
    """PyTorch view over Hugging Face CIFAR-10 records."""

    def __init__(self, dataset: HuggingFaceDataset) -> None:
        super().__init__(
            dataset,
            transform=_CIFAR10_IMAGE_TRANSFORM,
            image_column=IMAGE_COLUMN,
        )


def create_partitions(
    labels: Sequence[int],
    *,
    num_partitions: int = 4,
    mode: PartitionMode = "homogeneous",
    seed: int = 42,
    partition_profile: str = "auto",
    client_weights: Sequence[float] | None = None,
) -> list[list[int]]:
    """Create balanced, quantity-skewed, or strongly label-skewed partitions."""
    if num_partitions < 1:
        raise ValueError("num_partitions must be positive.")
    if mode not in ("homogeneous", "non-iid", "label-skew"):
        raise ValueError("mode must be 'homogeneous', 'non-iid', or 'label-skew'.")
    if partition_profile.lower() not in ("auto", "scalable"):
        raise ValueError("CIFAR-10 supports 'auto' and 'scalable' profiles.")
    if client_weights is not None and mode != "non-iid":
        raise ValueError("client_weights are only supported for non-IID partitions.")

    label_array = np.asarray(labels, dtype=np.int64)
    if mode == "homogeneous":
        return balanced_stratified_partitions(
            label_array, num_partitions, seed=seed
        )
    if mode == "label-skew":
        return label_shard_partitions(label_array, num_partitions, seed=seed)
    return quantity_skewed_partitions(
        len(label_array),
        num_partitions,
        seed=seed,
        weights=client_weights,
    )


class Cifar10DataModule:
    """Federated data module for the full ten-class ``uoft-cs/cifar10``."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        train_fraction: float = 0.8,
    ) -> None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1).")
        self.cache_dir = cache_dir
        self.train_fraction = train_fraction
        self.class_names = CLASS_NAMES
        self._dataset: DatasetDict | None = None

    @property
    def dataset(self) -> DatasetDict:
        if self._dataset is None:
            self._dataset = load_cifar10_dataset(self.cache_dir)
        return self._dataset

    def client_loaders(
        self,
        partition_id: int,
        *,
        num_partitions: int,
        partition_mode: PartitionMode,
        batch_size: int,
        seed: int,
        partition_profile: str = "auto",
        client_weights: Sequence[float] | None = None,
        max_samples: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        split = self.dataset["train"]
        labels = labels_from_records(split)
        partitions = create_partitions(
            labels,
            num_partitions=num_partitions,
            mode=partition_mode,
            seed=seed,
            partition_profile=partition_profile,
            client_weights=client_weights,
        )
        if not 0 <= partition_id < len(partitions):
            raise ValueError("partition_id must be in [0, num_partitions).")
        return make_client_loaders(
            Cifar10Dataset(split),
            labels,
            partitions[partition_id],
            batch_size=batch_size,
            seed=seed + partition_id,
            train_fraction=self.train_fraction,
            max_samples=max_samples,
        )

    def server_loaders(
        self,
        *,
        batch_size: int,
        seed: int,
        max_samples: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        split = self.dataset["test"]
        return make_server_loaders(
            Cifar10Dataset(split),
            labels_from_records(split),
            batch_size=batch_size,
            seed=seed,
            validation_fraction=0.5,
            max_samples=max_samples,
        )


def create_data_module(config: Mapping[str, Any]) -> Cifar10DataModule:
    """Factory used by the configurable ClientApp and ServerApp."""
    cache_dir = str(config.get("data-cache-dir", "")).strip() or None
    return Cifar10DataModule(
        cache_dir=cache_dir,
        train_fraction=float(config.get("train-fraction", 0.8)),
    )
