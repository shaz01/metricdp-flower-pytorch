"""Tests for the generic N-client target-shadow-loader."""

from __future__ import annotations

import pytest
from datasets import DatasetDict

from experiments.cia_client_scaling.shadow import target_shadow_loader
from experiments.reproduce.dataset.alzheimer import create_partitions
from metricdp_pytorch.utils.data import labels_from_records
from metricdp_pytorch.utils.split_data import split_stratified


def test_shadow_loader_is_ten_percent_of_targets_train_indices(
    alzheimer_dataset: DatasetDict,
) -> None:
    labels = labels_from_records(alzheimer_dataset["train"])
    partitions = create_partitions(
        labels, num_partitions=48, mode="homogeneous", seed=42
    )
    target_indices = partitions[0]
    train_indices, _ = split_stratified(labels, target_indices, 0.8, seed=42 + 0)
    expected_shadow, _ = split_stratified(labels, train_indices, 0.10, seed=42)

    loader = target_shadow_loader(
        target_partition_id=0,
        num_partitions=48,
        partition_mode="homogeneous",
        batch_size=32,
        seed=42,
    )
    assert len(loader.dataset) == len(expected_shadow)
    assert len(loader.dataset) == pytest.approx(0.10 * len(train_indices), abs=2)
    resolved_shadow_indices = [
        train_indices[index] for index in loader.dataset.indices
    ]
    assert resolved_shadow_indices == list(expected_shadow)
    assert set(resolved_shadow_indices) <= set(train_indices)


def test_shadow_loader_is_deterministic(alzheimer_dataset: DatasetDict) -> None:
    loader_a = target_shadow_loader(
        target_partition_id=0,
        num_partitions=48,
        partition_mode="non-iid",
        batch_size=32,
        seed=42,
    )
    loader_b = target_shadow_loader(
        target_partition_id=0,
        num_partitions=48,
        partition_mode="non-iid",
        batch_size=32,
        seed=42,
    )
    assert list(loader_a.dataset.indices) == list(loader_b.dataset.indices)


def test_shadow_loader_rejects_out_of_range_target_partition_id(
    alzheimer_dataset: DatasetDict,
) -> None:
    with pytest.raises(ValueError, match="target_partition_id"):
        target_shadow_loader(
            target_partition_id=48,
            num_partitions=48,
            partition_mode="homogeneous",
            batch_size=32,
            seed=42,
        )
