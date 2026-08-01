"""Generic N-client target-shadow-loader for the CIA-at-48-clients experiment.

Generalizes ``experiments/cia/dataset.py``'s fixed 3-client
``target_shadow_loader`` to arbitrary ``num_partitions``/``partition_mode`` by
calling the same ``create_partitions`` primitive the default Alzheimer data
module already uses for training, so the shadow sample always matches what
the target client actually trained on.
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from experiments.reproduce.dataset.alzheimer import (
    AlzheimerMRIDataset,
    PartitionMode,
    create_partitions,
    load_alzheimer_dataset,
)
from metricdp_pytorch.utils.data import cap_indices, labels_from_records, make_indexed_loader
from metricdp_pytorch.utils.split_data import split_stratified

SHADOW_FRACTION = 0.10
TRAIN_FRACTION = 0.8


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
    """The attacker's shadow set: a stratified 10% of the target client's
    train indices (Section 7.4.1). This overlaps with, and is not excluded
    from, what the target actually trains on."""
    if not 0 <= target_partition_id < num_partitions:
        raise ValueError(
            f"target_partition_id must be in [0, {num_partitions})."
        )
    dataset = load_alzheimer_dataset(cache_dir)
    split = dataset["train"]
    labels = labels_from_records(split)
    partitions = create_partitions(
        labels, num_partitions=num_partitions, mode=partition_mode, seed=seed
    )
    target_indices = cap_indices(partitions[target_partition_id], max_samples)
    train_indices, _test_indices = split_stratified(
        labels, target_indices, TRAIN_FRACTION, seed=seed + target_partition_id
    )
    shadow_indices, _rest = split_stratified(
        labels, train_indices, SHADOW_FRACTION, seed=seed
    )
    return make_indexed_loader(
        AlzheimerMRIDataset(split),
        shadow_indices,
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )
