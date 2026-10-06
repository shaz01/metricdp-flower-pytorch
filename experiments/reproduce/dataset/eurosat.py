"""EuroSAT data plugin -- satellite land-use imagery, 10 classes, RGB 64x64.

A genuinely different domain from every other dataset plugin in this repo (Alzheimer MRI,
CIFAR-10, CIFAR-100, Fashion-MNIST) -- chosen as a simpler comparison point for the accuracy
sweep methodology developed for CIFAR-100 (see results/contest_at_scale/cifar100/), avoiding both
CIFAR-10 (a teammate's part of the project) and the MNIST family. Like CIFAR-100, all classes
are used -- EuroSAT only has 10 to begin with, no subsetting needed.

See results/contest_at_scale/eurosat/ for the experiment that uses it.
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
    dirichlet_label_partitions,
    label_shard_partitions,
    quantity_skewed_partitions,
    split_stratified,
)

DATASET_ID = "tanganke/eurosat"
IMAGE_SIZE = (64, 64)
IMAGE_COLUMN = "image"
LABEL_COLUMN = "label"
_TO_TENSOR = ToTensor()
_RANDOM_CROP = RandomCrop(IMAGE_SIZE, padding=4)
_RANDOM_HORIZONTAL_FLIP = RandomHorizontalFlip(0.5)


def load_eurosat_dataset(cache_dir: str | Path | None = None) -> DatasetDict:
    """Return the full, unfiltered 10-class EuroSAT dataset."""
    return load_hf_dataset_cached(DATASET_ID, cache_dir)


def derive_class_names(train_split: HuggingFaceDataset) -> tuple[str, ...]:
    """Return class names from ``label``'s ClassLabel feature if present,
    otherwise synthesize numbered names from the observed label range.
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
    # Downsample the validated 64x64 image to this size (e.g. (32, 32)) so the
    # same 32x32 model can run on EuroSAT and CIFAR-10. None keeps native size.
    resize_to: tuple[int, int] | None = None

    def __call__(self, image: Any) -> torch.Tensor:
        if not isinstance(image, Image.Image):
            raise TypeError("The image column must decode to a PIL image.")
        rgb = image.convert("RGB")
        if rgb.size != self.image_size:
            raise ValueError(
                f"Expected {self.image_size[0]}×{self.image_size[1]} images, "
                f"got {rgb.size}."
            )
        if self.resize_to is not None and self.resize_to != self.image_size:
            rgb = rgb.resize(self.resize_to, Image.Resampling.BILINEAR)
        if self.augment:
            rgb = (_RANDOM_CROP if rgb.size == IMAGE_SIZE else RandomCrop(rgb.size[::-1], padding=4))(rgb)
            rgb = _RANDOM_HORIZONTAL_FLIP(rgb)
        return _TO_TENSOR(rgb)


_EUROSAT_TRAIN_TRANSFORM = RGBImageTransform(IMAGE_SIZE, augment=True)
_EUROSAT_EVAL_TRANSFORM = RGBImageTransform(IMAGE_SIZE, augment=False)


class EurosatDataset(RecordImageDataset):
    """PyTorch view over Hugging Face EuroSAT records."""

    def __init__(
        self,
        dataset: HuggingFaceDataset,
        *,
        augment: bool = False,
        resize_to: tuple[int, int] | None = None,
    ) -> None:
        if resize_to is None:
            transform = _EUROSAT_TRAIN_TRANSFORM if augment else _EUROSAT_EVAL_TRANSFORM
        else:
            transform = RGBImageTransform(IMAGE_SIZE, augment=augment, resize_to=resize_to)
        super().__init__(
            dataset,
            transform=transform,
            image_column=IMAGE_COLUMN,
            label_column=LABEL_COLUMN,
        )


def create_partitions(
    labels: Sequence[int],
    *,
    num_partitions: int = 48,
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
        raise ValueError("EuroSAT supports 'auto' and 'scalable' profiles.")
    if client_weights is not None and mode != "non-iid":
        raise ValueError("client_weights are only supported for non-IID partitions.")

    label_array = np.asarray(labels, dtype=np.int64)
    if mode == "homogeneous":
        return balanced_stratified_partitions(label_array, num_partitions, seed=seed)
    if mode == "label-skew":
        return label_shard_partitions(label_array, num_partitions, seed=seed)
    if mode == "dirichlet":
        return dirichlet_label_partitions(
            label_array, num_partitions, seed=seed, alpha=dirichlet_alpha
        )
    return quantity_skewed_partitions(
        len(label_array), num_partitions, seed=seed, weights=client_weights
    )


class EurosatDataModule:
    """Federated data module for full 10-class ``tanganke/eurosat``."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        train_fraction: float = 0.8,
        augment: bool = True,
        resize_to: tuple[int, int] | None = None,
        train_subsample: int = 0,
        subsample_seed: int = 0,
    ) -> None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1).")
        if resize_to is not None and (len(resize_to) != 2 or min(resize_to) < 1):
            raise ValueError("resize_to must be a (width, height) pair of positive ints.")
        if train_subsample < 0:
            raise ValueError("train_subsample must be non-negative (0 = full training set).")
        self.cache_dir = cache_dir
        self.train_fraction = train_fraction
        # Stratified subset of the training split taken BEFORE partitioning (same
        # semantics as Cifar10DataModule.train_subsample); 0 keeps the full split.
        self.train_subsample = train_subsample
        self.subsample_seed = subsample_seed
        self._train_pool: HuggingFaceDataset | None = None
        self.augment = augment  # training-time random crop + flip (default on)
        self.resize_to = None if resize_to is None else tuple(int(v) for v in resize_to)
        self._dataset: DatasetDict | None = None
        self._class_names: tuple[str, ...] | None = None

    def _view(self, split: HuggingFaceDataset, *, train: bool) -> EurosatDataset:
        return EurosatDataset(split, augment=train and self.augment, resize_to=self.resize_to)

    @property
    def dataset(self) -> DatasetDict:
        if self._dataset is None:
            self._dataset = load_eurosat_dataset(self.cache_dir)
        return self._dataset

    @property
    def class_names(self) -> tuple[str, ...]:
        if self._class_names is None:
            self._class_names = derive_class_names(self.dataset["train"])
        return self._class_names

    @property
    def train_pool(self) -> HuggingFaceDataset:
        """Training records partitioned across clients (full split or stratified subset)."""
        if self._train_pool is None:
            split = self.dataset["train"]
            if self.train_subsample and self.train_subsample < len(split):
                labels = labels_from_records(split, label_column=LABEL_COLUMN)
                keep, _ = split_stratified(
                    labels, range(len(labels)), self.train_subsample / len(labels),
                    seed=self.subsample_seed,
                )
                split = split.select(sorted(keep))
            self._train_pool = split
        return self._train_pool

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
        split = self.train_pool
        labels = labels_from_records(split, label_column=LABEL_COLUMN)
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

        loader_seed = seed + partition_id
        selected = cap_indices(partitions[partition_id], max_samples)
        train_indices, test_indices = split_stratified(
            labels, selected, self.train_fraction, seed=loader_seed
        )
        train_loader = make_indexed_loader(
            self._view(split, train=True),
            train_indices,
            batch_size=batch_size,
            shuffle=True,
            seed=loader_seed,
        )
        eval_loader = make_indexed_loader(
            self._view(split, train=False),
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
                labels, all_indices, max_samples / len(all_indices), seed=seed
            )
        else:
            selected = all_indices
        validation_indices, test_indices = split_stratified(
            labels, selected, 0.5, seed=seed
        )
        validation_loader = make_indexed_loader(
            self._view(split, train=False),
            validation_indices,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        test_loader = make_indexed_loader(
            self._view(split, train=False),
            test_indices,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        return validation_loader, test_loader


def create_data_module(config: Mapping[str, Any]) -> EurosatDataModule:
    """Factory used by the configurable ClientApp and ServerApp."""
    cache_dir = str(config.get("data-cache-dir", "")).strip() or None
    return EurosatDataModule(
        cache_dir=cache_dir,
        train_fraction=float(config.get("train-fraction", 0.8)),
    )
