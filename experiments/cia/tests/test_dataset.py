"""Tests for generic and paper-exact shadow data modules."""

from __future__ import annotations

import numpy as np
import pytest
from datasets import DatasetDict

from experiments.cia.datasets.paper import (
    PAPER_CIA_CLIENT_COUNTS,
    PAPER_CIA_NUM_CLIENTS,
    PAPER_CIA_TARGET_PARTITION_ID,
    PaperShadowDataModule,
    create_paper_shadow_data_module,
)
from experiments.cia.datasets.shadow import ShadowDataModule
from experiments.reproduce.dataset.alzheimer import AlzheimerDataModule
from metricdp_pytorch.utils.data import labels_from_records

PAPER_TRAIN_CLASS_COUNTS = (724, 49, 2566, 1781)
PAPER_TRAIN_TOTAL = 5120


def _inject_dataset(module: ShadowDataModule, dataset: DatasetDict) -> None:
    module.data_module._dataset = dataset  # type: ignore[attr-defined]


def test_client_row_sums_match_published_totals() -> None:
    published_totals = (1747, 1491, 1882)
    for row, total in zip(PAPER_CIA_CLIENT_COUNTS, published_totals, strict=True):
        assert sum(row) == total


def test_corrected_column_sums_match_table_1_train_counts() -> None:
    columns = tuple(
        sum(row[class_index] for row in PAPER_CIA_CLIENT_COUNTS)
        for class_index in range(4)
    )
    assert columns == PAPER_TRAIN_CLASS_COUNTS
    assert sum(sum(row) for row in PAPER_CIA_CLIENT_COUNTS) == PAPER_TRAIN_TOTAL


def test_paper_partitions_are_deterministic_and_match_class_counts(
    alzheimer_dataset: DatasetDict,
) -> None:
    module = PaperShadowDataModule()
    _inject_dataset(module, alzheimer_dataset)
    labels = labels_from_records(alzheimer_dataset["train"])

    partitions_a = module.paper_data_module.partitions(seed=42)
    partitions_b = module.paper_data_module.partitions(seed=42)
    assert partitions_a == partitions_b

    label_array = np.asarray(labels)
    for partition, expected_counts in zip(
        partitions_a, PAPER_CIA_CLIENT_COUNTS, strict=True
    ):
        observed = tuple(np.bincount(label_array[partition], minlength=4).tolist())
        assert observed == expected_counts


def test_paper_shadow_is_subset_of_targets_actual_train_dataset(
    alzheimer_dataset: DatasetDict,
) -> None:
    module = PaperShadowDataModule()
    _inject_dataset(module, alzheimer_dataset)
    target_train, _ = module.paper_data_module.client_loaders(
        PAPER_CIA_TARGET_PARTITION_ID,
        num_partitions=PAPER_CIA_NUM_CLIENTS,
        partition_mode="homogeneous",
        batch_size=32,
        seed=42,
        partition_profile="exact",
    )

    shadow_loader = module.target_shadow_loader(batch_size=32, seed=42)

    assert shadow_loader.dataset.dataset.indices == target_train.dataset.indices
    assert len(shadow_loader.dataset) == pytest.approx(
        0.10 * len(target_train.dataset), abs=2
    )


@pytest.mark.parametrize("num_clients", (4, 8, 16, 48, 128))
def test_generic_shadow_decorator_supports_scalable_client_counts(
    alzheimer_dataset: DatasetDict, num_clients: int
) -> None:
    module = ShadowDataModule(
        AlzheimerDataModule(),
        num_clients=num_clients,
        target_partition_id=0,
        shadow_fraction=0.2,
        partition_mode="homogeneous",
    )
    _inject_dataset(module, alzheimer_dataset)
    target_train, _ = module.data_module.client_loaders(
        0,
        num_partitions=num_clients,
        partition_mode="homogeneous",
        batch_size=32,
        seed=42,
        partition_profile="auto",
    )

    shadow_loader = module.target_shadow_loader(batch_size=32, seed=42)

    assert shadow_loader.dataset.dataset.indices == target_train.dataset.indices
    assert len(shadow_loader.dataset) == pytest.approx(
        0.2 * len(target_train.dataset), abs=2
    )


def test_paper_factory_configures_shadow_and_train_fractions(
    alzheimer_dataset: DatasetDict,
) -> None:
    module = create_paper_shadow_data_module(
        {
            "num-clients": 3,
            "target-partition-id": 1,
            "shadow-fraction": 0.2,
            "train-fraction": 0.7,
        }
    )
    _inject_dataset(module, alzheimer_dataset)

    shadow_loader = module.target_shadow_loader(batch_size=32, seed=42)

    assert module.target_partition_id == 1
    assert module.shadow_fraction == 0.2
    assert module.paper_data_module.train_fraction == 0.7
    assert len(shadow_loader.dataset) > 0


def test_rejects_target_outside_configured_client_count() -> None:
    with pytest.raises(ValueError, match="target_partition_id"):
        ShadowDataModule(
            AlzheimerDataModule(), num_clients=4, target_partition_id=4
        )
