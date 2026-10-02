"""Fashion-MNIST data plugin for reproduction experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict
from torch.utils.data import DataLoader
from experiments.reproduce.dataset.common import (
    PartitionMode,
    grayscale_image_transform,
    load_hf_dataset_cached,
)
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

DATASET_ID = "zalando-datasets/fashion_mnist"
IMAGE_SIZE = (28, 28)
CLASS_NAMES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
)
CLASS_IDS = tuple(range(len(CLASS_NAMES)))


def load_fashion_mnist_dataset(
    cache_dir: str | Path | None = None,
) -> DatasetDict:
    """Return Fashion-MNIST filtered to the four supported classes."""
    dataset = load_hf_dataset_cached(DATASET_ID, cache_dir)
    return dataset.filter(
        lambda label: label in CLASS_IDS,
        input_columns=["label"],
        desc="Filtering Fashion-MNIST to four classes",
    )


_FASHION_MNIST_IMAGE_TRANSFORM = grayscale_image_transform(IMAGE_SIZE)


class FashionMNISTDataset(RecordImageDataset):
    """PyTorch view over Hugging Face Fashion-MNIST records."""

    def __init__(self, dataset: HuggingFaceDataset) -> None:
        super().__init__(dataset, transform=_FASHION_MNIST_IMAGE_TRANSFORM)


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
        raise ValueError("Fashion-MNIST supports 'auto' and 'scalable' profiles.")
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


class FashionMNISTDataModule:
    """Federated data module for ``zalando-datasets/fashion_mnist``."""

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
            self._dataset = load_fashion_mnist_dataset(self.cache_dir)
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
            FashionMNISTDataset(split),
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
            FashionMNISTDataset(split),
            labels_from_records(split),
            batch_size=batch_size,
            seed=seed,
            validation_fraction=0.5,
            max_samples=max_samples,
        )


def create_data_module(config: Mapping[str, Any]) -> FashionMNISTDataModule:
    """Factory used by the configurable ClientApp and ServerApp."""
    cache_dir = str(config.get("data-cache-dir", "")).strip() or None
    return FashionMNISTDataModule(
        cache_dir=cache_dir,
        train_fraction=float(config.get("train-fraction", 0.8)),
    )
