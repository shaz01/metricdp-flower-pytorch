"""Tests for EuroSAT multi-round CIA experiment construction."""

from __future__ import annotations

import pytest

from experiments.cia.scripts.eurosat_scaling import (
    AGGREGATION,
    DEFAULT_OUTPUT_DIR,
    NOISE_STD_FRACTION,
    NUM_CLIENTS,
    PARTITION_MODES,
    PRIVACY_MODES,
    SEEDS,
    SHADOW_FRACTION,
    TARGET_PARTITION_ID,
    build_combos,
    create_eurosat_in,
    create_eurosat_out,
)


def test_in_view_includes_every_canonical_client() -> None:
    view = create_eurosat_in({})

    assert view.canonical_num_partitions == NUM_CLIENTS
    assert view.active_partition_ids == tuple(range(NUM_CLIENTS))
    assert view.num_active_partitions == NUM_CLIENTS


def test_out_view_excludes_only_the_target() -> None:
    view = create_eurosat_out({})

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
        assert combo.noise_multiplier == pytest.approx(0.03710712210729851)
        assert combo.hyperparams.rounds == 100
        assert combo.hyperparams.clipping_norm == 5.0
        assert combo.hyperparams.local_epochs == 5
        assert combo.hyperparams.batch_size == 32
        assert combo.hyperparams.learning_rate == pytest.approx(0.001)
        assert combo.hyperparams.initialization_epochs == 20
        assert combo.model_module == "experiments.reproduce.eurosat_cnn:create_model"
        assert combo.aggregation == "fedavg"


def test_module_level_shadow_and_noise_std_fractions() -> None:
    assert SHADOW_FRACTION == 0.10
    assert NOISE_STD_FRACTION == 0.20


def test_module_level_aggregation_is_fedavg() -> None:
    assert AGGREGATION == "fedavg"


def test_combo_name_prefix_matches_group() -> None:
    for group in ("in", "out"):
        for combo in build_combos(group):
            assert combo.name_prefix == f"eurosat-{group}-remove"


def test_default_output_dir_targets_results_cia_eurosat_scaling() -> None:
    assert DEFAULT_OUTPUT_DIR.name == "cia_eurosat_scaling"
    assert DEFAULT_OUTPUT_DIR.parent.name == "results"


def test_combos_wire_the_correct_data_module_per_group() -> None:
    for combo in build_combos("in"):
        assert (
            combo.data_module
            == "experiments.cia.scripts.eurosat_scaling:create_eurosat_in"
        )
    for combo in build_combos("out"):
        assert (
            combo.data_module
            == "experiments.cia.scripts.eurosat_scaling:create_eurosat_out"
        )


def test_build_combos_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match='"in" or "out"'):
        build_combos("sideways")


def test_checkpoint_rounds_cover_round_1_and_every_tenth_through_100() -> None:
    from experiments.cia.scripts.eurosat_scaling import CHECKPOINT_ROUNDS

    assert CHECKPOINT_ROUNDS[0] == 1
    assert CHECKPOINT_ROUNDS[-1] == 100
    assert len(CHECKPOINT_ROUNDS) == 11
    assert CHECKPOINT_ROUNDS == (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
