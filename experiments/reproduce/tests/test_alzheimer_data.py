"""Verify that the data pipeline exactly reproduces paper Tables 1–3."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from datasets import DatasetDict

from experiments.reproduce.dataset.alzheimer import (
    CLASS_NAMES,
    PAPER_HOMOGENEOUS_CLIENT_COUNTS,
    PAPER_NON_IID_CLIENT_COUNTS,
    PAPER_TEST_CLASS_COUNTS,
    PAPER_TRAIN_CLASS_COUNTS,
    AlzheimerMRIDataset,
    create_partitions,
    load_server_data,
    partition_train_indices,
)


def _class_counts(labels: np.ndarray, indices: list[int] | None = None) -> tuple[int, ...]:
    selected = labels if indices is None else labels[indices]
    return tuple(np.bincount(selected, minlength=4).tolist())


def test_table_1_original_train_and_test_distribution(
    alzheimer_dataset: DatasetDict,
) -> None:
    """The downloaded HF data must match the paper's original split table."""
    train_labels = np.asarray(alzheimer_dataset["train"]["label"], dtype=np.int64)
    test_labels = np.asarray(alzheimer_dataset["test"]["label"], dtype=np.int64)

    assert len(train_labels) == 5120
    assert _class_counts(train_labels) == PAPER_TRAIN_CLASS_COUNTS
    assert len(test_labels) == 1280
    assert _class_counts(test_labels) == PAPER_TEST_CLASS_COUNTS
    assert tuple(alzheimer_dataset["train"].features["label"].names) == CLASS_NAMES


@pytest.mark.parametrize(
    ("mode", "expected_rows"),
    [
        ("homogeneous", PAPER_HOMOGENEOUS_CLIENT_COUNTS),
        ("non-iid", PAPER_NON_IID_CLIENT_COUNTS),
    ],
)
def test_tables_2_and_3_client_distributions(
    alzheimer_dataset: DatasetDict,
    mode: str,
    expected_rows: tuple[tuple[int, int, int, int], ...],
) -> None:
    """Each generated client must have the exact published class counts."""
    labels = np.asarray(alzheimer_dataset["train"]["label"], dtype=np.int64)
    partitions = [
        partition_train_indices(
            labels,
            partition_id=client_id,
            num_partitions=4,
            mode=mode,
            seed=42,
        )
        for client_id in range(4)
    ]

    for indices, expected_counts in zip(partitions, expected_rows, strict=True):
        assert len(indices) == sum(expected_counts)
        assert _class_counts(labels, indices) == expected_counts

    flattened = [index for indices in partitions for index in indices]
    assert len(flattened) == 5120
    assert len(set(flattened)) == 5120
    assert sorted(flattened) == list(range(5120))


def test_homogeneous_distribution_scales_to_128_clients(
    alzheimer_dataset: DatasetDict,
) -> None:
    """Custom homogeneous runs remain balanced, disjoint, and stratified."""
    labels = np.asarray(alzheimer_dataset["train"]["label"], dtype=np.int64)
    partitions = [
        partition_train_indices(
            labels,
            partition_id=client_id,
            num_partitions=128,
            mode="homogeneous",
            seed=42,
        )
        for client_id in range(128)
    ]

    assert {len(indices) for indices in partitions} == {40}
    flattened = [index for indices in partitions for index in indices]
    assert len(flattened) == len(set(flattened)) == 5120
    class_counts = np.asarray([_class_counts(labels, indices) for indices in partitions])
    assert np.all(class_counts.max(axis=0) - class_counts.min(axis=0) <= 1)


def test_exact_paper_profile_rejects_other_client_counts(
    alzheimer_dataset: DatasetDict,
) -> None:
    labels = np.asarray(alzheimer_dataset["train"]["label"], dtype=np.int64)
    with pytest.raises(ValueError, match="requires four clients"):
        create_partitions(
            labels,
            num_partitions=128,
            mode="homogeneous",
            partition_profile="exact",
            seed=42,
        )


def test_non_iid_distribution_scales_to_128_clients(
    alzheimer_dataset: DatasetDict,
) -> None:
    """Default scalable quantity skew creates nonempty, unequal clients."""
    labels = np.asarray(alzheimer_dataset["train"]["label"], dtype=np.int64)
    partitions = [
        partition_train_indices(
            labels,
            partition_id=client_id,
            num_partitions=128,
            mode="non-iid",
            seed=42,
        )
        for client_id in range(128)
    ]

    sizes = [len(indices) for indices in partitions]
    assert min(sizes) > 0
    assert min(sizes) < max(sizes)
    flattened = [index for indices in partitions for index in indices]
    assert len(flattened) == len(set(flattened)) == 5120


def test_custom_non_iid_client_weights_control_partition_sizes(
    alzheimer_dataset: DatasetDict,
) -> None:
    labels = np.asarray(alzheimer_dataset["train"]["label"], dtype=np.int64)
    weights = [1.0, 2.0, 3.0, 4.0]
    partitions = [
        partition_train_indices(
            labels,
            partition_id=client_id,
            num_partitions=4,
            mode="non-iid",
            seed=42,
            exact_paper_distribution=False,
            client_weights=weights,
        )
        for client_id in range(4)
    ]

    assert [len(indices) for indices in partitions] == [512, 1024, 1536, 2048]


def test_server_test_split_is_stratified_50_50(
    alzheimer_dataset: DatasetDict,
) -> None:
    """The official test split becomes 640 validation and 640 final-test rows."""
    validation_loader, test_loader = load_server_data(
        batch_size=32,
        seed=42,
        dataset=alzheimer_dataset["test"],
    )
    labels = np.asarray(alzheimer_dataset["test"]["label"], dtype=np.int64)
    validation_indices = list(validation_loader.dataset.indices)
    test_indices = list(test_loader.dataset.indices)

    assert len(validation_indices) == 640
    assert len(test_indices) == 640
    assert set(validation_indices).isdisjoint(test_indices)
    assert set(validation_indices) | set(test_indices) == set(range(1280))
    assert _class_counts(labels, validation_indices) == (86, 8, 317, 229)
    assert _class_counts(labels, test_indices) == (86, 7, 317, 230)


def test_mri_adapter_returns_paper_model_input(
    alzheimer_dataset: DatasetDict,
) -> None:
    """Decoded HF images become one-channel 128×128 float tensors."""
    image, label = AlzheimerMRIDataset(alzheimer_dataset["train"])[0]

    assert image.shape == (1, 128, 128)
    assert image.dtype == torch.float32
    assert 0.0 <= float(image.min()) <= float(image.max()) <= 1.0
    assert label in range(4)
