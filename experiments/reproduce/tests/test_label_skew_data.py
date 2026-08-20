"""Cross-dataset integration tests for the strong label-skew partition mode."""

from __future__ import annotations

from collections import Counter

import pytest

from experiments.reproduce.dataset.alzheimer import create_partitions as alzheimer_partitions
from experiments.reproduce.dataset.cifar4 import create_partitions as cifar4_partitions
from experiments.reproduce.dataset.cifar10 import create_partitions as cifar10_partitions
from experiments.reproduce.dataset.cifar100 import create_partitions as cifar100_partitions
from experiments.reproduce.dataset.eurosat import create_partitions as eurosat_partitions
from experiments.reproduce.dataset.fashion_mnist import (
    create_partitions as fashion_partitions,
)


@pytest.mark.parametrize(
    "create_partitions",
    [
        alzheimer_partitions,
        cifar4_partitions,
        cifar10_partitions,
        cifar100_partitions,
        eurosat_partitions,
        fashion_partitions,
    ],
)
def test_dataset_plugins_expose_strong_label_skew(create_partitions) -> None:
    labels = [label for label in range(4) for _ in range(100)]

    partitions = create_partitions(
        labels,
        num_partitions=8,
        mode="label-skew",
        seed=42,
    )

    assert sorted(index for partition in partitions for index in partition) == list(
        range(len(labels))
    )
    sizes = [len(partition) for partition in partitions]
    assert max(sizes) - min(sizes) <= 4
    assert all(
        len(Counter(labels[index] for index in partition)) <= 4
        for partition in partitions
    )


def test_alzheimer_label_skew_does_not_select_four_client_paper_profile() -> None:
    """The new mode must work at n=4 even though profile=auto is paper-exact there."""
    labels = [label for label in range(4) for _ in range(100)]

    partitions = alzheimer_partitions(
        labels,
        num_partitions=4,
        mode="label-skew",
        seed=42,
    )

    assert [len(partition) for partition in partitions] == [100] * 4
    assert all(
        len({labels[index] for index in partition}) <= 4
        for partition in partitions
    )
