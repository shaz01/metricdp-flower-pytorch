"""Full 100-class CIFAR-100 data plugin.

Unlike every other dataset plugin in this repo (cifar10.py, fashion_mnist.py,
alzheimer.py), which all subset to four classes to match the paper's model,
this one keeps all 100 fine-grained classes. See
experiments/cifar100_scaling/ for the experiment that uses it.
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
from torchvision.transforms import RandomCrop, RandomHorizontalFlip, ToTensor

from experiments.reproduce.dataset.common import PartitionMode, load_hf_dataset_cached
from metricdp_pytorch.utils.data import (
    RecordImageDataset,
    cap_indices,
    labels_from_records,
    make_indexed_loader,
)
from metricdp_pytorch.utils.split_data import (
    balanced_stratified_partitions,
    quantity_skewed_partitions,
    split_stratified,
)

DATASET_ID = "uoft-cs/cifar100"
IMAGE_SIZE = (32, 32)
IMAGE_COLUMN = "img"
LABEL_COLUMN = "fine_label"
_TO_TENSOR = ToTensor()
_RANDOM_CROP = RandomCrop(IMAGE_SIZE, padding=4)
_RANDOM_HORIZONTAL_FLIP = RandomHorizontalFlip(0.5)


def load_cifar100_dataset(cache_dir: str | Path | None = None) -> DatasetDict:
    """Return the full, unfiltered 100-class CIFAR-100 dataset."""
    return load_hf_dataset_cached(DATASET_ID, cache_dir)


def derive_class_names(train_split: HuggingFaceDataset) -> tuple[str, ...]:
    """Return class names from ``fine_label``'s ClassLabel feature if present,
    otherwise synthesize numbered names from the observed label range.

    The real Hugging Face ``fine_label`` column is a ``ClassLabel`` feature
    with real names; a plain ``Dataset.from_dict`` test double (int64 column,
    no ``ClassLabel``) falls back to synthesized names instead of crashing.
    """
    feature = train_split.features.get(LABEL_COLUMN)
    names = getattr(feature, "names", None)
    if names:
        return tuple(names)
    labels = train_split[LABEL_COLUMN]
    num_classes = int(max(labels)) + 1 if labels else 0
    return tuple(f"class_{index}" for index in range(num_classes))


@dataclass(frozen=True)
class RGBImageTransform:
    """Pickle-safe validated RGB tensor transform, optionally augmented."""

    image_size: tuple[int, int]
    augment: bool = False

    def __call__(self, image: Any) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("The image column must decode to a PIL image.")
        rgb = image.convert("RGB")
        if rgb.size != self.image_size:
            raise ValueError(
                f"Expected {self.image_size[0]}×{self.image_size[1]} images, "
                f"got {rgb.size}."
            )
        if self.augment:
            rgb = _RANDOM_CROP(rgb)
            rgb = _RANDOM_HORIZONTAL_FLIP(rgb)
        return _TO_TENSOR(rgb)


_CIFAR100_TRAIN_TRANSFORM = RGBImageTransform(IMAGE_SIZE, augment=True)
_CIFAR100_EVAL_TRANSFORM = RGBImageTransform(IMAGE_SIZE, augment=False)


class Cifar100Dataset(RecordImageDataset):
    """PyTorch view over Hugging Face CIFAR-100 records."""

    def __init__(self, dataset: HuggingFaceDataset, *, augment: bool = False) -> None:
        super().__init__(
            dataset,
            transform=_CIFAR100_TRAIN_TRANSFORM if augment else _CIFAR100_EVAL_TRANSFORM,
            image_column=IMAGE_COLUMN,
            label_column=LABEL_COLUMN,
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
    """Create balanced or quantity-skewed CIFAR-100 partitions."""
    if num_partitions < 1:
        raise ValueError("num_partitions must be positive.")
    if mode not in ("homogeneous", "non-iid"):
        raise ValueError("mode must be 'homogeneous' or 'non-iid'.")
    if partition_profile.lower() not in ("auto", "scalable"):
        raise ValueError("CIFAR-100 supports 'auto' and 'scalable' profiles.")
    if client_weights is not None and mode != "non-iid":
        raise ValueError("client_weights are only supported for non-IID partitions.")

    label_array = np.asarray(labels, dtype=np.int64)
    if mode == "homogeneous":
        return balanced_stratified_partitions(
            label_array, num_partitions, seed=seed
        )
    return quantity_skewed_partitions(
        len(label_array),
        num_partitions,
        seed=seed,
        weights=client_weights,
    )


class Cifar100DataModule:
    """Federated data module for full 100-class ``uoft-cs/cifar100``."""

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
        self._dataset: DatasetDict | None = None
        self._class_names: tuple[str, ...] | None = None

    @property
    def dataset(self) -> DatasetDict:
        if self._dataset is None:
            self._dataset = load_cifar100_dataset(self.cache_dir)
        return self._dataset

    @property
    def class_names(self) -> tuple[str, ...]:
        if self._class_names is None:
            self._class_names = derive_class_names(self.dataset["train"])
        return self._class_names

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
        labels = labels_from_records(split, label_column=LABEL_COLUMN)
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

        loader_seed = seed + partition_id
        selected = cap_indices(partitions[partition_id], max_samples)
        train_indices, test_indices = split_stratified(
            labels, selected, self.train_fraction, seed=loader_seed
        )
        train_loader = make_indexed_loader(
            Cifar100Dataset(split, augment=True),
            train_indices,
            batch_size=batch_size,
            shuffle=True,
            seed=loader_seed,
        )
        eval_loader = make_indexed_loader(
            Cifar100Dataset(split, augment=False),
            test_indices,
            batch_size=batch_size,
            shuffle=False,
            seed=loader_seed,
        )
        return train_loader, eval_loader

    def server_loaders(
        self,
        *,
        batch_size: int,
        seed: int,
        max_samples: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        split = self.dataset["test"]
        labels = labels_from_records(split, label_column=LABEL_COLUMN)
        all_indices = list(range(len(labels)))
        if max_samples < 0:
            raise ValueError("max_samples must be non-negative.")
        if 0 < max_samples < len(all_indices):
            selected, _ = split_stratified(
                labels,
                all_indices,
                max_samples / len(all_indices),
                seed=seed,
            )
        else:
            selected = all_indices
        validation_indices, test_indices = split_stratified(
            labels, selected, 0.5, seed=seed
        )
        validation_loader = make_indexed_loader(
            Cifar100Dataset(split, augment=False),
            validation_indices,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        test_loader = make_indexed_loader(
            Cifar100Dataset(split, augment=False),
            test_indices,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        return validation_loader, test_loader


def create_data_module(config: Mapping[str, Any]) -> Cifar100DataModule:
    """Factory used by the configurable ClientApp and ServerApp."""
    cache_dir = str(config.get("data-cache-dir", "")).strip() or None
    return Cifar100DataModule(
        cache_dir=cache_dir,
        train_fraction=float(config.get("train-fraction", 0.8)),
    )
