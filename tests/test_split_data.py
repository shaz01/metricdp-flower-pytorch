"""Tests for deterministic dataset splitting and partitioning helpers."""

from collections import Counter

import pytest

from metricdp_pytorch.utils.split_data import partition_by_class_counts


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
