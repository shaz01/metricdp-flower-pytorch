"""Tests for CIA participant partition views."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

from experiments.cia.datasets.partitions import (
    PartitionViewDataModule,
    in_remove,
    in_replace,
    out_remove,
    out_replace,
)
from metricdp_pytorch.utils.data import make_client_loaders
from metricdp_pytorch.utils.split_data import balanced_stratified_partitions


class RecordingDataModule:
    def __init__(self) -> None:
        self.client_calls: list[tuple[int, int]] = []
        self.server_calls: list[tuple[int, int, int]] = []
        dataset = TensorDataset(torch.zeros(1), torch.zeros(1, dtype=torch.long))
        self.loader = DataLoader(dataset, batch_size=1)
        self.class_names = ("zero", "one")

    def client_loaders(
        self,
        partition_id: int,
        *,
        num_partitions: int,
        partition_mode: str,
        batch_size: int,
        seed: int,
        partition_profile: str = "auto",
        client_weights: Sequence[float] | None = None,
        dirichlet_alpha: float = 0.5,
        max_samples: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        del partition_mode, batch_size, seed, partition_profile, client_weights
        del dirichlet_alpha, max_samples
        self.client_calls.append((partition_id, num_partitions))
        return self.loader, self.loader

    def server_loaders(
        self, *, batch_size: int, seed: int, max_samples: int = 0
    ) -> tuple[DataLoader, DataLoader]:
        self.server_calls.append((batch_size, seed, max_samples))
        return self.loader, self.loader


class SeededDataModule:
    """Synthetic module whose canonical partition membership depends on seed."""

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
        client_weights: Sequence[float] | None = None,
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

def _load_every_active_partition(view: PartitionViewDataModule) -> None:
    for partition_id in range(view.num_active_partitions):
        view.client_loaders(
            partition_id,
            num_partitions=view.num_active_partitions,
            partition_mode="homogeneous",
            batch_size=32,
            seed=42,
        )


def _active_train_indices(
    view: PartitionViewDataModule, *, seed: int
) -> dict[int, tuple[int, ...]]:
    """Return source indices keyed by canonical partition ID."""
    memberships: dict[int, tuple[int, ...]] = {}
    for active_id in range(view.num_active_partitions):
        train_loader, _ = view.client_loaders(
            active_id,
            num_partitions=view.num_active_partitions,
            partition_mode="homogeneous",
            batch_size=8,
            seed=seed,
        )
        assert isinstance(train_loader.dataset, Subset)
        memberships[view.canonical_partition_id(active_id)] = tuple(
            int(index) for index in train_loader.dataset.indices
        )
    return memberships


ViewFactory = Callable[[SeededDataModule], PartitionViewDataModule]


@pytest.mark.parametrize(
    "view_factory",
    (
        pytest.param(
            lambda module: in_remove(
                module, canonical_num_partitions=4, target_partition_id=2
            ),
            id="in-remove",
        ),
        pytest.param(
            lambda module: out_remove(
                module, canonical_num_partitions=4, target_partition_id=2
            ),
            id="out-remove",
        ),
        pytest.param(
            lambda module: in_replace(
                module,
                canonical_num_partitions=4,
                target_partition_id=2,
                replacement_partition_id=3,
            ),
            id="in-replace",
        ),
        pytest.param(
            lambda module: out_replace(
                module,
                canonical_num_partitions=4,
                target_partition_id=2,
                replacement_partition_id=3,
            ),
            id="out-replace",
        ),
    ),
)
def test_partition_views_are_deterministic_with_seed(
    view_factory: ViewFactory,
) -> None:
    view = view_factory(SeededDataModule())

    seed_42_a = _active_train_indices(view, seed=42)
    seed_42_b = _active_train_indices(view, seed=42)
    seed_43 = _active_train_indices(view, seed=43)

    assert seed_42_a == seed_42_b
    assert seed_42_a != seed_43


def test_remove_views_include_or_exclude_target_without_repartitioning() -> None:
    in_data = RecordingDataModule()
    out_data = RecordingDataModule()

    in_view = in_remove(
        in_data, canonical_num_partitions=3, target_partition_id=2
    )
    out_view = out_remove(
        out_data, canonical_num_partitions=3, target_partition_id=2
    )
    _load_every_active_partition(in_view)
    _load_every_active_partition(out_view)

    assert in_view.active_partition_ids == (0, 1, 2)
    assert out_view.active_partition_ids == (0, 1)
    assert in_data.client_calls == [(0, 3), (1, 3), (2, 3)]
    assert out_data.client_calls == [(0, 3), (1, 3)]


def test_replace_views_swap_target_and_replacement_at_fixed_client_count() -> None:
    in_data = RecordingDataModule()
    out_data = RecordingDataModule()

    in_view = in_replace(
        in_data,
        canonical_num_partitions=4,
        target_partition_id=2,
        replacement_partition_id=3,
    )
    out_view = out_replace(
        out_data,
        canonical_num_partitions=4,
        target_partition_id=2,
        replacement_partition_id=3,
    )
    _load_every_active_partition(in_view)
    _load_every_active_partition(out_view)

    assert in_view.active_partition_ids == (0, 1, 2)
    assert out_view.active_partition_ids == (0, 1, 3)
    assert in_data.client_calls == [(0, 4), (1, 4), (2, 4)]
    assert out_data.client_calls == [(0, 4), (1, 4), (3, 4)]
    assert in_view.num_active_partitions == out_view.num_active_partitions == 3


def test_view_requires_flower_to_use_active_partition_count() -> None:
    view = out_remove(
        RecordingDataModule(),
        canonical_num_partitions=3,
        target_partition_id=2,
    )

    with pytest.raises(ValueError, match="number of active partitions"):
        view.client_loaders(
            0,
            num_partitions=3,
            partition_mode="homogeneous",
            batch_size=32,
            seed=42,
        )


def test_server_loading_is_delegated_unchanged() -> None:
    data_module = RecordingDataModule()
    view = out_remove(
        data_module, canonical_num_partitions=3, target_partition_id=2
    )

    view.server_loaders(batch_size=16, seed=7, max_samples=10)

    assert data_module.server_calls == [(16, 7, 10)]
    assert view.class_names == ("zero", "one")


def test_replacement_target_and_alternative_must_differ() -> None:
    with pytest.raises(ValueError, match="must differ"):
        in_replace(
            RecordingDataModule(),
            canonical_num_partitions=4,
            target_partition_id=2,
            replacement_partition_id=2,
        )
