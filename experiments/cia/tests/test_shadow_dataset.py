"""Tests for shadow datasets built from a combo's training data factory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from experiments.cia.datasets.partitions import out_remove
from experiments.cia.scripts import contest
from experiments.cia.shadow_dataset import clean_shadow_dataset, noisy_shadow_dataset
from experiments.reproduce.dataset.alzheimer import (
    AlzheimerDataModule,
    create_data_module as create_alzheimer_data_module,
)
from experiments.reproduce.matrix import Combo, Hyperparams
from metricdp_pytorch.utils.noisy_dataset import NoisyDataModule


def create_out_remove_test_data_module(config: Mapping[str, Any]):
    """Script-style factory used to verify canonical target access."""
    active_count = int(config["num-clients"])
    return out_remove(
        create_alzheimer_data_module(config),
        canonical_num_partitions=active_count + 1,
        target_partition_id=0,
    )


def _out_remove_combo() -> Combo:
    return Combo(
        name_prefix="cia-out-remove",
        num_clients=3,
        partition="homogeneous",
        privacy="vanilla",
        aggregation="fedavg",
        seed=42,
        noise_multiplier=0.01,
        hyperparams=Hyperparams(
            clipping_norm=5.0,
            rounds=20,
            local_epochs=5,
            batch_size=32,
            learning_rate=0.001,
            initialization_epochs=20,
        ),
        data_module=(
            "experiments.cia.tests.test_shadow_dataset:"
            "create_out_remove_test_data_module"
        ),
        model_module="experiments.reproduce.paper_cnn:create_model",
    )


def test_contest_matrix_uses_its_local_participant_factory() -> None:
    view = contest.create_data_module({"num-clients": 4})

    assert contest.MATRIX.data_module == (
        "experiments.cia.scripts.contest:create_data_module"
    )
    assert view.canonical_num_partitions == 4
    assert view.active_partition_ids == (0, 1, 2, 3)


def test_shadow_datasets_use_canonical_data_behind_training_view() -> None:
    combo = _out_remove_combo()

    clean = clean_shadow_dataset(
        combo, target_partition_id=0, shadow_fraction=0.1
    )
    noisy = noisy_shadow_dataset(
        combo,
        target_partition_id=0,
        shadow_fraction=0.1,
        std_fraction=0.2,
    )

    assert clean.num_clients == 4
    assert clean.target_partition_id == 0
    assert isinstance(clean.data_module, AlzheimerDataModule)
    assert noisy.num_clients == 4
    assert noisy.target_partition_id == 0
    assert isinstance(noisy.data_module, NoisyDataModule)
    assert isinstance(noisy.data_module.data_module, AlzheimerDataModule)
    assert noisy.data_module.std_fraction == 0.2
