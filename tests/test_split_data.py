"""Tests for deterministic dataset splitting and partitioning helpers."""

from collections import Counter

import pytest

from metricdp_pytorch.utils.split_data import (
    label_shard_partitions,
    partition_by_class_counts,
)


def test_label_shards_are_balanced_complete_deterministic_and_skewed() -> None:
    labels = [label for label in range(8) for _ in range(100)]

    seed_42_a = label_shard_partitions(labels, 8, seed=42)
    seed_42_b = label_shard_partitions(labels, 8, seed=42)
    seed_43 = label_shard_partitions(labels, 8, seed=43)

    assert seed_42_a == seed_42_b
    assert seed_42_a != seed_43
    assert sorted(index for partition in seed_42_a for index in partition) == list(
        range(len(labels))
    )
    assert {len(partition) for partition in seed_42_a} == {100}
    assert all(
        len({labels[index] for index in partition}) <= 4
        for partition in seed_42_a
    )


def test_label_shards_support_uneven_classes_and_shard_sizes() -> None:
    labels = [0] * 9 + [1] * 7 + [2] * 5

    partitions = label_shard_partitions(
        labels, 5, seed=42, shards_per_partition=2
    )

    assert sorted(index for partition in partitions for index in partition) == list(
        range(len(labels))
    )
    assert max(map(len, partitions)) - min(map(len, partitions)) <= 2


@pytest.mark.parametrize(
    ("labels", "num_partitions", "shards_per_partition", "message"),
    [
        ([0, 1], 0, 2, "num_partitions"),
        ([0, 1], 1, 0, "shards_per_partition"),
        ([0, 1, 2], 2, 2, "at least one example"),
    ],
)
def test_label_shards_reject_invalid_configuration(
    labels, num_partitions, shards_per_partition, message
) -> None:
    with pytest.raises(ValueError, match=message):
        label_shard_partitions(
            labels,
            num_partitions,
            seed=42,
            shards_per_partition=shards_per_partition,
        )


def test_class_count_partition_can_leave_seeded_surplus_unused() -> None:
    labels = [0] * 20 + [1] * 20
    counts = ((3, 2), (2, 4))

    seed_42_a = partition_by_class_counts(labels, counts, seed=42)
    seed_42_b = partition_by_class_counts(labels, counts, seed=42)
    seed_43 = partition_by_class_counts(labels, counts, seed=43)

    assert seed_42_a == seed_42_b
    assert seed_42_a != seed_43
    assert len(set(seed_42_a[0]) & set(seed_42_a[1])) == 0
    assert Counter(labels[index] for index in seed_42_a[0]) == {0: 3, 1: 2}
    assert Counter(labels[index] for index in seed_42_a[1]) == {0: 2, 1: 4}
    assert sum(map(len, seed_42_a)) == 11 < len(labels)


def test_class_count_partition_rejects_insufficient_class_records() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        partition_by_class_counts(
            [0, 0, 1, 1],
            ((3, 1),),
            seed=42,
        )
