"""Tests for chunked CIFAR PLAN.md experiment construction."""

import pytest

from experiments.cia.scripts.cifar_chunks import (
    build_combos,
    create_cifar_in_replace,
    create_cifar_out_remove_all,
    create_cifar_out_replace,
)


def test_max_record_chunk_has_two_active_clients_and_fixed_multiplier() -> None:
    combos = build_combos(
        task="max-records",
        adjacency="out-remove",
        active_clients=2,
        seed=42,
        privacy_modes=("global-dp", "metric-privacy"),
        noise_ratio=None,
    )

    assert [combo.privacy for combo in combos] == ["global-dp", "metric-privacy"]
    assert {combo.num_clients for combo in combos} == {2}
    assert {combo.noise_multiplier for combo in combos} == {0.01}


def test_replacement_multiplier_is_ratio_times_active_clients() -> None:
    combos = build_combos(
        task="noise-sweep",
        adjacency="in-replace",
        active_clients=3,
        seed=42,
        privacy_modes=("global-dp", "metric-privacy"),
        noise_ratio=0.003333,
    )

    assert all(
        combo.noise_multiplier == pytest.approx(0.009999) for combo in combos
    )


def test_replacement_views_use_one_extra_canonical_partition() -> None:
    config = {"num-clients": 48}
    in_view = create_cifar_in_replace(config)
    out_view = create_cifar_out_replace(config)

    assert in_view.canonical_num_partitions == 49
    assert in_view.active_partition_ids == tuple(range(48))
    assert out_view.canonical_num_partitions == 49
    assert out_view.active_partition_ids == tuple(range(1, 49))


def test_max_record_view_partitions_all_records_before_removing_target() -> None:
    view = create_cifar_out_remove_all({})

    assert view.canonical_num_partitions == 3
    assert view.active_partition_ids == (1, 2)


def test_non_vanilla_replacement_chunk_requires_ratio() -> None:
    with pytest.raises(ValueError, match="positive ratio"):
        build_combos(
            task="full",
            adjacency="out-replace",
            active_clients=8,
            seed=43,
            privacy_modes=("global-dp",),
            noise_ratio=None,
        )
