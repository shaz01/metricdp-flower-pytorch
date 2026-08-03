"""Tests for reusable dataset utilities."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from metricdp_pytorch.data_module import FederatedDataModule
from metricdp_pytorch.utils.data import make_indexed_loader
from metricdp_pytorch.utils.noisy_dataset import NoisyDataModule, NoisyDataset


def test_indexed_loader_uses_parallel_prefetching() -> None:
    dataset = TensorDataset(torch.ones(4, 3), torch.tensor([0, 1, 0, 1]))

    loader = make_indexed_loader(
        dataset,
        range(4),
        batch_size=2,
        shuffle=True,
        seed=9,
    )

    assert loader.num_workers == 2
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 2
    assert loader.pin_memory is torch.cuda.is_available()


def test_noisy_dataset_is_seeded_and_preserves_labels() -> None:
    base = TensorDataset(torch.ones(2, 3), torch.tensor([0, 1]))
    noisy = NoisyDataset(base, std_fraction=0.2, seed=9)

    first, first_label = noisy[0]
    repeated, repeated_label = noisy[0]

    assert torch.equal(first, repeated)
    assert not torch.equal(first, base[0][0])
    assert first_label == repeated_label == 0


def test_noisy_dataset_rejects_negative_noise() -> None:
    base = TensorDataset(torch.ones(1, 3), torch.tensor([0]))

    with pytest.raises(ValueError, match="non-negative"):
        NoisyDataset(base, std_fraction=-0.1, seed=9)


def test_noisy_data_module_wraps_all_returned_datasets() -> None:
    dataset = TensorDataset(torch.ones(4, 3), torch.tensor([0, 1, 0, 1]))

    class DataModule:
        class_names = ("zero", "one")

        def client_loaders(self, partition_id: int, **_kwargs):
            return DataLoader(dataset, batch_size=2), DataLoader(dataset, batch_size=2)

        def server_loaders(self, **_kwargs):
            return DataLoader(dataset, batch_size=2), DataLoader(dataset, batch_size=2)

    module = NoisyDataModule(DataModule(), std_fraction=0.2)

    assert isinstance(module, FederatedDataModule)
    assert module.class_names == ("zero", "one")
    client_loaders = module.client_loaders(
        0,
        num_partitions=1,
        partition_mode="homogeneous",
        batch_size=2,
        seed=9,
    )
    server_loaders = module.server_loaders(batch_size=2, seed=9)

    for loader in (*client_loaders, *server_loaders):
        assert isinstance(loader.dataset, NoisyDataset)
        values, labels = next(iter(loader))
        assert not torch.equal(values, torch.ones_like(values))
        assert labels.tolist() == [0, 1]
