"""Four-class CIFAR-10 data plugin for reproduction experiments."""

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
    dirichlet_label_partitions,
    label_shard_partitions,
    quantity_skewed_partitions,
)

DATASET_ID = "uoft-cs/cifar10"
IMAGE_SIZE = (32, 32)
IMAGE_COLUMN = "img"
CLASS_NAMES = ("airplane", "automobile", "bird", "cat")
CLASS_IDS = tuple(range(len(CLASS_NAMES)))
_TO_TENSOR = ToTensor()


def load_cifar4_dataset(cache_dir: str | Path | None = None) -> DatasetDict:
    """Return CIFAR-10 filtered to the first four classes."""
    dataset = load_hf_dataset_cached(DATASET_ID, cache_dir)
    return dataset.filter(
        lambda label: label in CLASS_IDS,
        input_columns=["label"],
        desc="Filtering CIFAR-10 to four classes",
    )


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


_CIFAR4_IMAGE_TRANSFORM = RGBImageTransform(IMAGE_SIZE)


class Cifar4Dataset(RecordImageDataset):
    """PyTorch view over Hugging Face CIFAR-10 records."""

    def __init__(self, dataset: HuggingFaceDataset) -> None:
        super().__init__(
            dataset,
            transform=_CIFAR4_IMAGE_TRANSFORM,
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
    dirichlet_alpha: float = 0.5,
) -> list[list[int]]:
    """Create balanced, quantity-skewed, shard-skewed, or Dirichlet partitions."""
    if num_partitions < 1:
        raise ValueError("num_partitions must be positive.")
    if mode not in ("homogeneous", "non-iid", "label-skew", "dirichlet"):
        raise ValueError(
            "mode must be 'homogeneous', 'non-iid', 'label-skew', or 'dirichlet'."
        )
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
    if mode == "dirichlet":
        return dirichlet_label_partitions(
            label_array, num_partitions, seed=seed, alpha=dirichlet_alpha
        )
    return quantity_skewed_partitions(
        len(label_array),
        num_partitions,
        seed=seed,
        weights=client_weights,
    )


class Cifar4DataModule:
    """Federated data module for four-class ``uoft-cs/cifar10``."""

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
            self._dataset = load_cifar4_dataset(self.cache_dir)
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
        dirichlet_alpha: float = 0.5,
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
            dirichlet_alpha=dirichlet_alpha,
        )
        if not 0 <= partition_id < len(partitions):
            raise ValueError("partition_id must be in [0, num_partitions).")
        return make_client_loaders(
            Cifar4Dataset(split),
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
            Cifar4Dataset(split),
            labels_from_records(split),
            batch_size=batch_size,
            seed=seed,
            validation_fraction=0.5,
            max_samples=max_samples,
        )


def create_data_module(config: Mapping[str, Any]) -> Cifar4DataModule:
    """Factory used by the configurable ClientApp and ServerApp."""
    cache_dir = str(config.get("data-cache-dir", "")).strip() or None
    return Cifar4DataModule(
        cache_dir=cache_dir,
        train_fraction=float(config.get("train-fraction", 0.8)),
    )
