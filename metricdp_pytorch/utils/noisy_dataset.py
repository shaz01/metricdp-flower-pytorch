"""Deterministic noisy dataset and federated data-module decorators."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset

from metricdp_pytorch.data_module import FederatedDataModule
from metricdp_pytorch.utils.data import Sample


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


def _with_noisy_dataset(
    loader: DataLoader,
    *,
    std_fraction: float,
    seed: int,
) -> DataLoader:
    """Clone a loader while decorating its dataset and retaining its batching."""
    return DataLoader(
        NoisyDataset(loader.dataset, std_fraction=std_fraction, seed=seed),
        batch_sampler=loader.batch_sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers,
        pin_memory_device=loader.pin_memory_device,
    )


class NoisyDataModule:
    """Decorate every dataset returned by a federated data module with noise."""

    def __init__(
        self,
        data_module: FederatedDataModule,
        *,
        std_fraction: float,
    ) -> None:
        if std_fraction < 0:
            raise ValueError("std_fraction must be non-negative.")
        self.data_module = data_module
        self.std_fraction = std_fraction

    @property
    def class_names(self) -> Sequence[str]:
        """Expose class names from the wrapped data module when available."""
        return getattr(self.data_module, "class_names", ())

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
        train_loader, test_loader = self.data_module.client_loaders(
            partition_id,
            num_partitions=num_partitions,
            partition_mode=partition_mode,
            batch_size=batch_size,
            seed=seed,
            partition_profile=partition_profile,
            client_weights=client_weights,
            max_samples=max_samples,
        )
        return (
            _with_noisy_dataset(
                train_loader,
                std_fraction=self.std_fraction,
                seed=seed,
            ),
            _with_noisy_dataset(
                test_loader,
                std_fraction=self.std_fraction,
                seed=seed,
            ),
        )

    def server_loaders(
        self,
        *,
        batch_size: int,
        seed: int,
        max_samples: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        validation_loader, test_loader = self.data_module.server_loaders(
            batch_size=batch_size,
            seed=seed,
            max_samples=max_samples,
        )
        return (
            _with_noisy_dataset(
                validation_loader,
                std_fraction=self.std_fraction,
                seed=seed,
            ),
            _with_noisy_dataset(
                test_loader,
                std_fraction=self.std_fraction,
                seed=seed,
            ),
        )


# Explicit alias for callers that prefer the protocol-qualified name.
NoisyFederatedDataModule = NoisyDataModule
