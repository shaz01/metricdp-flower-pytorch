"""Dataset-independent deterministic splitting and partitioning helpers."""

from collections.abc import Sequence

import numpy as np


def split_total_by_weights(total: int, weights: Sequence[float]) -> list[int]:
    values = np.asarray(weights, dtype=np.float64)
    if total < 0 or np.any(values < 0) or values.sum() <= 0:
        raise ValueError("Weights must be non-negative and have a positive sum.")
    raw = total * values / values.sum()
    counts = np.floor(raw).astype(int)
    largest_remainders = np.argsort(-(raw - counts), kind="stable")
    counts[largest_remainders[: total - int(counts.sum())]] += 1
    return counts.tolist()


def balanced_stratified_partitions(
    labels: Sequence[int],
    num_partitions: int,
    *,
    seed: int,
) -> list[list[int]]:
    """Distribute labelled examples into balanced, stratified partitions.

    Every class is divided as evenly as possible. Remainders are assigned to
    the currently smallest partitions with seeded random tie-breaking, which
    keeps total partition sizes balanced without favoring low partition IDs.
    """
    if num_partitions < 1:
        raise ValueError("num_partitions must be positive.")

    label_array = np.asarray(labels)
    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[] for _ in range(num_partitions)]
    partition_totals = np.zeros(num_partitions, dtype=np.int64)
    for label in np.unique(label_array):
        class_indices = np.flatnonzero(label_array == label)
        rng.shuffle(class_indices)
        base_count, remainder = divmod(len(class_indices), num_partitions)
        class_counts = np.full(num_partitions, base_count, dtype=np.int64)
        tie_breakers = rng.random(num_partitions)
        priority = np.lexsort((tie_breakers, partition_totals))
        class_counts[priority[:remainder]] += 1

        offset = 0
        for partition_id, count in enumerate(class_counts):
            partitions[partition_id].extend(
                class_indices[offset : offset + count].tolist()
            )
            partition_totals[partition_id] += count
            offset += count

    for partition in partitions:
        rng.shuffle(partition)
    return partitions


def partition_by_class_counts(
    labels: Sequence[int],
    counts_by_partition: Sequence[Sequence[int]],
    *,
    seed: int,
) -> list[list[int]]:
    """Partition examples using an exact partition-by-class count matrix.

    Matrix rows correspond to partitions and columns to sorted unique labels.
    Every example must be consumed exactly once.
    """
    label_array = np.asarray(labels)
    unique_labels = np.unique(label_array)
    count_matrix = np.asarray(counts_by_partition, dtype=np.int64)
    if count_matrix.ndim != 2 or count_matrix.shape[1] != len(unique_labels):
        raise ValueError(
            "counts_by_partition must have one column per unique label."
        )
    if count_matrix.shape[0] < 1 or np.any(count_matrix < 0):
        raise ValueError("The count matrix must be non-negative and non-empty.")
    observed = np.asarray(
        [np.sum(label_array == label) for label in unique_labels], dtype=np.int64
    )
    if not np.array_equal(count_matrix.sum(axis=0), observed):
        raise ValueError(
            "Count-matrix column totals must match the observed class counts."
        )

    rng = np.random.default_rng(seed)
    partitions: list[list[int]] = [[] for _ in range(count_matrix.shape[0])]
    for column, label in enumerate(unique_labels):
        class_indices = np.flatnonzero(label_array == label)
        rng.shuffle(class_indices)
        offset = 0
        for partition_id, count in enumerate(count_matrix[:, column]):
            partitions[partition_id].extend(
                class_indices[offset : offset + count].tolist()
            )
            offset += int(count)
    for partition in partitions:
        rng.shuffle(partition)
    return partitions


def weighted_random_partitions(
    indices: Sequence[int],
    weights: Sequence[float],
    *,
    seed: int,
) -> list[list[int]]:
    """Shuffle indices and allocate them to quantity-weighted partitions."""
    if len(weights) < 1:
        raise ValueError("At least one partition weight is required.")
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    counts = split_total_by_weights(len(shuffled), weights)
    boundaries = np.cumsum([0, *counts])
    return [
        shuffled[boundaries[index] : boundaries[index + 1]].tolist()
        for index in range(len(weights))
    ]


def quantity_skewed_partitions(
    num_examples: int,
    num_partitions: int,
    *,
    seed: int,
    weights: Sequence[float] | None = None,
) -> list[list[int]]:
    """Create deterministic quantity-skewed partitions of ``range(num_examples)``."""
    if num_examples < 0:
        raise ValueError("num_examples must be non-negative.")
    if num_partitions < 1:
        raise ValueError("num_partitions must be positive.")
    rng = np.random.default_rng(seed)
    if weights is None:
        selected_weights = np.linspace(0.4, 1.6, num_partitions)
        rng.shuffle(selected_weights)
    else:
        if len(weights) != num_partitions:
            raise ValueError("weights length must equal num_partitions.")
        selected_weights = np.asarray(weights, dtype=np.float64)
    return weighted_random_partitions(
        range(num_examples), selected_weights, seed=seed
    )


def split_stratified(
        labels: Sequence[int],
        indices: Sequence[int],
        first_fraction: float,
        *,
        seed: int,
) -> tuple[list[int], list[int]]:
    """Return deterministic stratified first/rest subsets of ``indices``."""
    if not 0.0 < first_fraction < 1.0:
        raise ValueError("first_fraction must be strictly between zero and one.")
    label_array = np.asarray(labels)
    selected_indices = np.asarray(indices, dtype=np.int64)
    if len(selected_indices) < 2:
        raise ValueError("At least two selected indices are required.")
    if np.any(selected_indices < 0) or np.any(selected_indices >= len(label_array)):
        raise IndexError("indices contain values outside the labels sequence.")
    rng = np.random.default_rng(seed)
    first: list[int] = []
    rest: list[int] = []
    class_indices_by_label = [
        selected_indices[label_array[selected_indices] == label].copy()
        for label in np.unique(label_array[selected_indices])
    ]
    first_counts = split_total_by_weights(
        int(first_fraction * len(selected_indices)),
        [len(class_indices) for class_indices in class_indices_by_label],
    )
    for class_indices, split_at in zip(
            class_indices_by_label, first_counts, strict=True
    ):
        rng.shuffle(class_indices)
        first.extend(class_indices[:split_at].tolist())
        rest.extend(class_indices[split_at:].tolist())
    rng.shuffle(first)
    rng.shuffle(rest)
    return first, rest
