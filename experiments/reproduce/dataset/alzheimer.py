"""Alzheimer-MRI data plugin for the paper reproduction.

Only paper-specific dataset identity, image preprocessing, and published count
profiles live here. Generic partitioning, capping, stratified splitting, and
DataLoader construction live under ``metricdp_pytorch.utils``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from datasets import Dataset as HuggingFaceDataset
from datasets import DatasetDict, load_dataset
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from metricdp_pytorch.utils.data import (
    RecordImageDataset,
    labels_from_records,
    make_client_loaders,
    make_server_loaders,
)
from metricdp_pytorch.utils.split_data import (
    balanced_stratified_partitions,
    partition_by_class_counts,
    quantity_skewed_partitions,
)

DATASET_ID = "Falah/Alzheimer_MRI"
IMAGE_SIZE = (128, 128)
CLASS_NAMES = (
    "Mild_Demented",
    "Moderate_Demented",
    "Non_Demented",
    "Very_Mild_Demented",
)
PartitionMode = Literal["homogeneous", "non-iid"]

# Exact distributions reported in Tables 1–3. Rows correspond to clients 1–4;
# columns correspond to classes 0–3.
PAPER_TRAIN_CLASS_COUNTS = (724, 49, 2566, 1781)
PAPER_TEST_CLASS_COUNTS = (172, 15, 634, 459)
PAPER_HOMOGENEOUS_CLIENT_COUNTS = (
    (181, 12, 641, 446),
    (181, 12, 642, 445),
    (181, 12, 642, 445),
    (181, 13, 641, 445),
)
PAPER_NON_IID_CLIENT_COUNTS = (
    (280, 16, 881, 615),
    (107, 13, 368, 280),
    (257, 17, 1054, 720),
    (80, 3, 263, 166),
)

_TO_TENSOR = ToTensor()


_HF_OFFLINE_VARS = ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")


@contextmanager
def _hf_offline() -> Iterator[None]:
    """Temporarily pin HuggingFace lookups to the local cache."""
    previous = {name: os.environ.get(name) for name in _HF_OFFLINE_VARS}
    os.environ.update({name: "1" for name in _HF_OFFLINE_VARS})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@cache
def _load_alzheimer_dataset_once(cache_dir: str | None) -> DatasetDict:
    """Materialise one shared dataset instance for a resolved cache location."""
    if any(os.environ.get(name) is not None for name in _HF_OFFLINE_VARS):
        return load_dataset(DATASET_ID, cache_dir=cache_dir)
    try:
        with _hf_offline():
            return load_dataset(DATASET_ID, cache_dir=cache_dir)
    except Exception:  # noqa: BLE001 - any cache miss should retry over network
        return load_dataset(DATASET_ID, cache_dir=cache_dir)


def load_alzheimer_dataset(cache_dir: str | Path | None = None) -> DatasetDict:
    """Return the process-wide Alzheimer MRI dataset singleton.

    The first call for each cache location checks the local HuggingFace cache
    before allowing a network download. The resulting ``DatasetDict`` is then
    reused by every data-module instance in this process, avoiding repeated Hub
    checks when Flower rebuilds a data module for each client call. Separate
    worker processes still load their own instance, but do so from the shared
    on-disk HuggingFace cache.

    Explicit ``HF_*_OFFLINE`` settings are respected on the first call.
    """
    resolved_cache_dir = None if cache_dir is None else str(cache_dir)
    return _load_alzheimer_dataset_once(resolved_cache_dir)


def _alzheimer_image_transform(image: Any) -> torch.Tensor:
    """Convert one paper MRI image to a grayscale ``(1, 128, 128)`` tensor."""
    if not isinstance(image, Image.Image):
        raise TypeError("The image column must decode to a PIL image.")
    grayscale = image.convert("L")
    if grayscale.size != IMAGE_SIZE:
        raise ValueError(
            f"Expected {IMAGE_SIZE[0]}×{IMAGE_SIZE[1]} images, got {grayscale.size}."
        )
    return _TO_TENSOR(grayscale)


class AlzheimerMRIDataset(RecordImageDataset):
    """Paper-specific view over the generic record-image adapter."""

    def __init__(self, dataset: HuggingFaceDataset) -> None:
        super().__init__(dataset, transform=_alzheimer_image_transform)


def _use_exact_profile(profile: str, num_partitions: int) -> bool:
    normalized = profile.lower()
    if normalized == "auto":
        return num_partitions == 4
    if normalized == "exact":
        if num_partitions != 4:
            raise ValueError("The exact paper distribution requires four clients.")
        return True
    if normalized == "scalable":
        return False
    raise ValueError("partition_profile must be 'auto', 'exact', or 'scalable'.")


def create_partitions(
    labels: Sequence[int],
    *,
    num_partitions: int = 4,
    mode: PartitionMode = "homogeneous",
    seed: int = 42,
    partition_profile: str = "auto",
    client_weights: Sequence[float] | None = None,
) -> list[list[int]]:
    """Create all paper-profile or scalable client partitions."""
    if num_partitions < 1:
        raise ValueError("num_partitions must be positive.")
    if mode not in ("homogeneous", "non-iid"):
        raise ValueError("mode must be 'homogeneous' or 'non-iid'.")
    if client_weights is not None and mode != "non-iid":
        raise ValueError("client_weights are only supported for non-IID partitions.")

    label_array = np.asarray(labels, dtype=np.int64)
    if _use_exact_profile(partition_profile, num_partitions):
        observed = tuple(np.bincount(label_array, minlength=4).tolist())
        if observed != PAPER_TRAIN_CLASS_COUNTS:
            raise ValueError(
                f"Expected paper train counts {PAPER_TRAIN_CLASS_COUNTS}, got {observed}."
            )
        matrix = (
            PAPER_HOMOGENEOUS_CLIENT_COUNTS
            if mode == "homogeneous"
            else PAPER_NON_IID_CLIENT_COUNTS
        )
        return partition_by_class_counts(label_array, matrix, seed=seed)

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


def partition_train_indices(
    labels: Sequence[int],
    *,
    partition_id: int,
    num_partitions: int = 4,
    mode: PartitionMode = "homogeneous",
    seed: int = 42,
    exact_paper_distribution: bool | None = None,
    client_weights: Sequence[float] | None = None,
) -> list[int]:
    """Compatibility wrapper returning one partition from ``create_partitions``."""
    if not 0 <= partition_id < num_partitions:
        raise ValueError("partition_id must be in [0, num_partitions).")
    profile = (
        "auto"
        if exact_paper_distribution is None
        else "exact" if exact_paper_distribution else "scalable"
    )
    return create_partitions(
        labels,
        num_partitions=num_partitions,
        mode=mode,
        seed=seed,
        partition_profile=profile,
        client_weights=client_weights,
    )[partition_id]


class AlzheimerDataModule:
    """Pluggable data module for ``Falah/Alzheimer_MRI``."""

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

    @property
    def dataset(self) -> DatasetDict:
        if self._dataset is None:
            self._dataset = load_alzheimer_dataset(self.cache_dir)
        return self._dataset

    def client_loaders(
        self,
        partition_id: int,
        *,
        num_partitions: int,
        partition_mode: str,
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
            AlzheimerMRIDataset(split),
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
            AlzheimerMRIDataset(split),
            labels_from_records(split),
            batch_size=batch_size,
            seed=seed,
            validation_fraction=0.5,
            max_samples=max_samples,
        )


def create_data_module(config: Mapping[str, Any]) -> AlzheimerDataModule:
    """Factory used by the configurable ClientApp and ServerApp."""
    cache_dir = str(config.get("data-cache-dir", "")).strip() or None
    return AlzheimerDataModule(
        cache_dir=cache_dir,
        train_fraction=float(config.get("train-fraction", 0.8)),
    )


def load_client_data(
    partition_id: int,
    *,
    batch_size: int,
    mode: PartitionMode = "homogeneous",
    num_partitions: int = 4,
    seed: int = 42,
    exact_paper_distribution: bool | None = None,
    client_weights: Sequence[float] | None = None,
    max_samples: int = 0,
    dataset: HuggingFaceDataset | None = None,
    cache_dir: str | Path | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Compatibility function for direct tests and notebooks."""
    module = AlzheimerDataModule(cache_dir)
    if dataset is not None:
        module._dataset = DatasetDict({"train": dataset})
    # Direct injected train-only datasets cannot serve server loaders, which is
    # fine here because this compatibility function only requests client data.
    return module.client_loaders(
        partition_id,
        num_partitions=num_partitions,
        partition_mode=mode,
        batch_size=batch_size,
        seed=seed,
        partition_profile=(
            "auto"
            if exact_paper_distribution is None
            else "exact" if exact_paper_distribution else "scalable"
        ),
        client_weights=client_weights,
        max_samples=max_samples,
    )


def load_server_data(
    *,
    batch_size: int,
    seed: int = 42,
    dataset: HuggingFaceDataset | None = None,
    cache_dir: str | Path | None = None,
    max_samples: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Compatibility function for direct tests and notebooks."""
    module = AlzheimerDataModule(cache_dir)
    if dataset is not None:
        module._dataset = DatasetDict({"test": dataset})
    return module.server_loaders(
        batch_size=batch_size,
        seed=seed,
        max_samples=max_samples,
    )
