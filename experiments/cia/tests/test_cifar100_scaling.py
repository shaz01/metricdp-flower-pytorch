"""Tests for CIFAR-100 multi-round CIA experiment construction."""

from __future__ import annotations

from experiments.cia.scripts.cifar100_scaling import (
    NUM_CLIENTS,
    TARGET_PARTITION_ID,
    create_cifar100_in,
    create_cifar100_out,
)


def test_in_view_includes_every_canonical_client() -> None:
    view = create_cifar100_in({})

    assert view.canonical_num_partitions == NUM_CLIENTS
    assert view.active_partition_ids == tuple(range(NUM_CLIENTS))
    assert view.num_active_partitions == NUM_CLIENTS


def test_out_view_excludes_only_the_target() -> None:
    view = create_cifar100_out({})

    assert view.canonical_num_partitions == NUM_CLIENTS
    assert TARGET_PARTITION_ID not in view.active_partition_ids
    assert view.num_active_partitions == NUM_CLIENTS - 1
    assert view.active_partition_ids == tuple(
        partition_id
        for partition_id in range(NUM_CLIENTS)
        if partition_id != TARGET_PARTITION_ID
    )
