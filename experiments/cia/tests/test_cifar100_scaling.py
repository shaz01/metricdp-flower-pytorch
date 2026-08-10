"""Tests for CIFAR-100 multi-round CIA experiment construction."""

from __future__ import annotations

import pytest

from experiments.cia.scripts.cifar100_scaling import (
    NUM_CLIENTS,
    PARTITION_MODES,
    PRIVACY_MODES,
    SEEDS,
    TARGET_PARTITION_ID,
    build_combos,
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


def test_in_group_uses_all_clients_out_group_excludes_target() -> None:
    in_combos = build_combos("in")
    out_combos = build_combos("out")

    assert {combo.num_clients for combo in in_combos} == {NUM_CLIENTS}
    assert {combo.num_clients for combo in out_combos} == {NUM_CLIENTS - 1}


def test_build_combos_returns_one_per_partition_privacy_seed_triple() -> None:
    combos = build_combos("in")

    assert len(combos) == len(PARTITION_MODES) * len(PRIVACY_MODES) * len(SEEDS)
    assert {(combo.partition, combo.privacy, combo.seed) for combo in combos} == {
        (partition, privacy, seed)
        for partition in PARTITION_MODES
        for privacy in PRIVACY_MODES
        for seed in SEEDS
    }


def test_build_combos_covers_every_seed() -> None:
    combos = build_combos("out")

    assert {combo.seed for combo in combos} == set(SEEDS)
    assert SEEDS == (42, 43, 44)


def test_combos_share_the_sweep_hyperparameters() -> None:
    for combo in build_combos("in"):
        assert combo.seed in SEEDS
        assert combo.noise_multiplier == pytest.approx(0.0182)
        assert combo.hyperparams.rounds == 250
        assert combo.hyperparams.clipping_norm == 5.0
        assert combo.hyperparams.local_epochs == 5
        assert combo.hyperparams.weight_decay == pytest.approx(5e-4)
        assert combo.hyperparams.lr_schedule == "none"
        assert combo.model_module == "experiments.reproduce.cifar100_cnn:create_model"


def test_combos_wire_the_correct_data_module_per_group() -> None:
    for combo in build_combos("in"):
        assert (
            combo.data_module
            == "experiments.cia.scripts.cifar100_scaling:create_cifar100_in"
        )
    for combo in build_combos("out"):
        assert (
            combo.data_module
            == "experiments.cia.scripts.cifar100_scaling:create_cifar100_out"
        )


def test_build_combos_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match='"in" or "out"'):
        build_combos("sideways")
