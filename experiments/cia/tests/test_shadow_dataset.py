"""Tests for shadow datasets built from a combo's training data factory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from experiments.cia.datasets.partitions import out_remove
from experiments.cia.scripts import contest
from experiments.cia.shadow_dataset import clean_shadow_dataset, noisy_shadow_dataset
from experiments.reproduce.dataset.alzheimer import (
    AlzheimerDataModule,
    create_data_module as create_alzheimer_data_module,
)
from experiments.reproduce.matrix import Combo, Hyperparams
from metricdp_pytorch.utils.data import make_client_loaders
from metricdp_pytorch.utils.noisy_dataset import NoisyDataModule, NoisyDataset
from metricdp_pytorch.utils.split_data import balanced_stratified_partitions


class SeededTestDataModule:
    """Small data module exercising every seeded shadow-split layer."""

    def __init__(self) -> None:
        self.labels = [index % 4 for index in range(240)]
        self.dataset = TensorDataset(
            torch.arange(240, dtype=torch.float32).unsqueeze(1),
            torch.tensor(self.labels),
        )

    def client_loaders(
        self,
        partition_id: int,
        *,
        num_partitions: int,
        partition_mode: str,
        batch_size: int,
        seed: int,
        partition_profile: str = "auto",
        client_weights=None,
        dirichlet_alpha: float = 0.5,
        max_samples: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        del partition_mode, partition_profile, client_weights, dirichlet_alpha
        partitions = balanced_stratified_partitions(
            self.labels, num_partitions, seed=seed
        )
        return make_client_loaders(
            self.dataset,
            self.labels,
            partitions[partition_id],
            batch_size=batch_size,
            seed=seed + partition_id,
            max_samples=max_samples,
        )

    def server_loaders(
        self, *, batch_size: int, seed: int, max_samples: int = 0
    ) -> tuple[DataLoader, DataLoader]:
        return make_client_loaders(
            self.dataset,
            self.labels,
            list(range(len(self.dataset))),
            batch_size=batch_size,
            seed=seed,
            max_samples=max_samples,
        )


def create_seeded_test_data_module(_config: Mapping[str, Any]):
    return SeededTestDataModule()


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


def _seeded_combo(seed: int) -> Combo:
    return Combo(
        name_prefix="seeded-shadow-test",
        num_clients=4,
        partition="homogeneous",
        privacy="vanilla",
        aggregation="fedavg",
        seed=seed,
        noise_multiplier=0.01,
        hyperparams=Hyperparams(
            clipping_norm=5.0,
            rounds=1,
            local_epochs=1,
            batch_size=8,
            learning_rate=0.001,
            initialization_epochs=1,
        ),
        data_module=(
            "experiments.cia.tests.test_shadow_dataset:"
            "create_seeded_test_data_module"
        ),
        model_module="experiments.reproduce.paper_cnn:create_model",
    )


def _source_indices(loader: DataLoader) -> tuple[int, ...]:
    """Resolve a shadow subset back to the synthetic dataset's indices."""
    shadow_dataset = loader.dataset
    assert isinstance(shadow_dataset, Subset)
    target_train_dataset = shadow_dataset.dataset
    if isinstance(target_train_dataset, NoisyDataset):
        target_train_dataset = target_train_dataset.dataset
    assert isinstance(target_train_dataset, Subset)
    return tuple(
        int(target_train_dataset.indices[index])
        for index in shadow_dataset.indices
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


def test_shadow_dataset_helpers_repeat_membership_for_the_same_seed() -> None:
    combo = _seeded_combo(seed=42)

    clean_a = clean_shadow_dataset(
        combo, target_partition_id=0, shadow_fraction=0.25
    ).target_shadow_loader(batch_size=8, seed=combo.seed)
    clean_b = clean_shadow_dataset(
        combo, target_partition_id=0, shadow_fraction=0.25
    ).target_shadow_loader(batch_size=8, seed=combo.seed)
    noisy_a = noisy_shadow_dataset(
        combo,
        target_partition_id=0,
        shadow_fraction=0.25,
        std_fraction=0.2,
    ).target_shadow_loader(batch_size=8, seed=combo.seed)
    noisy_b = noisy_shadow_dataset(
        combo,
        target_partition_id=0,
        shadow_fraction=0.25,
        std_fraction=0.2,
    ).target_shadow_loader(batch_size=8, seed=combo.seed)

    clean_indices = _source_indices(clean_a)
    assert clean_indices == _source_indices(clean_b)
    assert clean_indices == _source_indices(noisy_a)
    assert clean_indices == _source_indices(noisy_b)
    assert all(
        torch.equal(noisy_a.dataset[index][0], noisy_b.dataset[index][0])
        for index in range(len(noisy_a.dataset))
    )


def test_shadow_dataset_helpers_change_membership_for_a_different_seed() -> None:
    combo_a = _seeded_combo(seed=42)
    combo_b = _seeded_combo(seed=43)

    clean_a = clean_shadow_dataset(
        combo_a, target_partition_id=0, shadow_fraction=0.25
    ).target_shadow_loader(batch_size=8, seed=combo_a.seed)
    clean_b = clean_shadow_dataset(
        combo_b, target_partition_id=0, shadow_fraction=0.25
    ).target_shadow_loader(batch_size=8, seed=combo_b.seed)
    noisy_a = noisy_shadow_dataset(
        combo_a, target_partition_id=0, shadow_fraction=0.25
    ).target_shadow_loader(batch_size=8, seed=combo_a.seed)
    noisy_b = noisy_shadow_dataset(
        combo_b, target_partition_id=0, shadow_fraction=0.25
    ).target_shadow_loader(batch_size=8, seed=combo_b.seed)

    assert _source_indices(clean_a) != _source_indices(clean_b)
    assert _source_indices(noisy_a) != _source_indices(noisy_b)
