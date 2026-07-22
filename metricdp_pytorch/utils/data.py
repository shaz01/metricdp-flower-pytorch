"""Dataset-independent PyTorch loader and dataset-adapter utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from metricdp_pytorch.utils.split_data import split_stratified

Sample = tuple[torch.Tensor, int]


class RecordImageDataset(Dataset[Sample]):
    """Adapt record-style image data to a transformed PyTorch dataset.

    The wrapped dataset only needs ``__len__``/``__getitem__`` and records with
    configurable image and label fields. This works with Hugging Face datasets
    but does not depend on one particular repository or image shape.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        transform: Callable[[Any], torch.Tensor],
        image_column: str = "image",
        label_column: str = "label",
    ) -> None:
        self.dataset = dataset
        self.transform = transform
        self.image_column = image_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        record = self.dataset[index]
        return self.transform(record[self.image_column]), int(record[self.label_column])


def make_indexed_loader(
    dataset: Dataset[Sample],
    indices: Sequence[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 2,
) -> DataLoader:
    """Create a deterministic, accelerator-friendly indexed DataLoader."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")
    selected = list(indices)
    if not selected:
        raise ValueError("Cannot create a loader for an empty index subset.")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, selected),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def cap_indices(indices: Sequence[int], max_samples: int) -> list[int]:
    """Apply a zero-means-unlimited deterministic sample cap."""
    if max_samples < 0:
        raise ValueError("max_samples must be non-negative.")
    selected = list(indices)
    return selected[:max_samples] if max_samples > 0 else selected


def make_client_loaders(
    dataset: Dataset[Sample],
    labels: Sequence[int],
    client_indices: Sequence[int],
    *,
    batch_size: int,
    seed: int,
    train_fraction: float = 0.8,
    max_samples: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Cap one client partition and make deterministic stratified loaders."""
    selected = cap_indices(client_indices, max_samples)
    train_indices, test_indices = split_stratified(
        labels, selected, train_fraction, seed=seed
    )
    return (
        make_indexed_loader(
            dataset,
            train_indices,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        ),
        make_indexed_loader(
            dataset,
            test_indices,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        ),
    )


def make_server_loaders(
    dataset: Dataset[Sample],
    labels: Sequence[int],
    *,
    batch_size: int,
    seed: int,
    validation_fraction: float = 0.5,
    max_samples: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Make stratified server validation and final-test loaders."""
    all_indices = list(range(len(dataset)))
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
        labels, selected, validation_fraction, seed=seed
    )
    return (
        make_indexed_loader(
            dataset,
            validation_indices,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        ),
        make_indexed_loader(
            dataset,
            test_indices,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        ),
    )


def labels_from_records(dataset: Any, label_column: str = "label") -> np.ndarray:
    """Read integer labels from a column-addressable record dataset."""
    return np.asarray(dataset[label_column], dtype=np.int64)


class NoisyDataset(Dataset[Sample]):
    """Add deterministic Gaussian noise to tensor samples from any dataset.

    ``std_fraction`` scales noise by each sample's maximum absolute value. The
    same index always receives the same noise for a given seed, making shadow
    and robustness experiments reproducible.
    """

    def __init__(self, dataset: Dataset[Sample], std_fraction: float, seed: int) -> None:
        if std_fraction < 0:
            raise ValueError("std_fraction must be non-negative.")
        self.dataset = dataset
        self.std_fraction = std_fraction
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Sample:
        value, label = self.dataset[index]
        if self.std_fraction == 0:
            return value, label
        generator = torch.Generator().manual_seed(self.seed + index)
        scale = self.std_fraction * float(value.abs().max())
        noise = torch.randn(value.shape, generator=generator, dtype=value.dtype)
        return value + scale * noise, label
