"""Tests for the paper's first-round CIA 3-client partition (Table 9, corrected)."""

from __future__ import annotations

import numpy as np
import pytest
from datasets import DatasetDict

from experiments.cia.dataset import (
    CIA_CLIENT_COUNTS,
    CIA_NUM_CLIENTS,
    CIA_SHADOW_FRACTION,
    CIA_TARGET_PARTITION_ID,
    CiaDataModule,
)
from metricdp_pytorch.utils.data import labels_from_records
from metricdp_pytorch.utils.split_data import split_stratified

PAPER_TRAIN_CLASS_COUNTS = (724, 49, 2566, 1781)
PAPER_TRAIN_TOTAL = 5120


def test_client_row_sums_match_published_totals() -> None:
    published_totals = (1747, 1491, 1882)
    for row, total in zip(CIA_CLIENT_COUNTS, published_totals, strict=True):
        assert sum(row) == total


def test_corrected_column_sums_match_table_1_train_counts() -> None:
    columns = tuple(
        sum(row[class_index] for row in CIA_CLIENT_COUNTS)
        for class_index in range(4)
    )
    assert columns == PAPER_TRAIN_CLASS_COUNTS
    assert sum(sum(row) for row in CIA_CLIENT_COUNTS) == PAPER_TRAIN_TOTAL


def test_num_clients_and_target_id_are_consistent() -> None:
    assert CIA_NUM_CLIENTS == len(CIA_CLIENT_COUNTS) == 3
    assert CIA_TARGET_PARTITION_ID == 2


def test_partitions_are_deterministic_and_match_class_counts(
    alzheimer_dataset: DatasetDict,
) -> None:
    module = CiaDataModule()
    module._dataset = alzheimer_dataset
    labels = labels_from_records(alzheimer_dataset["train"])

    partitions_a = module.partitions(seed=42)
    partitions_b = module.partitions(seed=42)
    assert partitions_a == partitions_b

    label_array = np.asarray(labels)
    for partition, expected_counts in zip(partitions_a, CIA_CLIENT_COUNTS, strict=True):
        observed = tuple(np.bincount(label_array[partition], minlength=4).tolist())
        assert observed == expected_counts


def test_shadow_loader_is_stratified_ten_percent_of_target_train_indices(
    alzheimer_dataset: DatasetDict,
) -> None:
    module = CiaDataModule()
    module._dataset = alzheimer_dataset
    labels = labels_from_records(alzheimer_dataset["train"])

    target_indices = module.partitions(seed=42)[CIA_TARGET_PARTITION_ID]
    train_indices, _ = split_stratified(
        labels, target_indices, 0.8, seed=42 + CIA_TARGET_PARTITION_ID
    )
    expected_shadow, _ = split_stratified(
        labels, train_indices, CIA_SHADOW_FRACTION, seed=42
    )

    shadow_loader = module.target_shadow_loader(batch_size=32, seed=42)
    assert len(shadow_loader.dataset) == len(expected_shadow)
    assert len(shadow_loader.dataset) == pytest.approx(0.10 * len(train_indices), abs=2)
