"""Compatibility helper backed by the generic CIA shadow decorator."""

from __future__ import annotations

from torch.utils.data import DataLoader

from experiments.cia.datasets.shadow import ShadowDataModule
from experiments.reproduce.dataset.alzheimer import AlzheimerDataModule, PartitionMode

SHADOW_FRACTION = 0.10


def target_shadow_loader(
    *,
    target_partition_id: int,
    num_partitions: int,
    partition_mode: PartitionMode,
    batch_size: int,
    seed: int,
    cache_dir: str | None = None,
    max_samples: int = 0,
) -> DataLoader:
    """Create the scaling experiment's shadow loader through composition."""
    module = ShadowDataModule(
        AlzheimerDataModule(cache_dir),
        num_clients=num_partitions,
        target_partition_id=target_partition_id,
        shadow_fraction=SHADOW_FRACTION,
        partition_mode=partition_mode,
        partition_profile="auto",
    )
    return module.target_shadow_loader(
        batch_size=batch_size, seed=seed, max_samples=max_samples
    )
