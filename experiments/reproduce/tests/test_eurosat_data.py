"""Tests for the EuroSAT data plugin."""

from __future__ import annotations

import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features
from PIL import Image

from experiments.reproduce.dataset import eurosat


def test_load_eurosat_dataset_keeps_all_classes(monkeypatch) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"label": list(range(10))}),
            "test": Dataset.from_dict({"label": list(reversed(range(10)))}),
        }
    )
    monkeypatch.setattr(
        eurosat,
        "load_hf_dataset_cached",
        lambda _dataset_id, _cache_dir: dataset,
    )

    loaded = eurosat.load_eurosat_dataset()

    assert loaded["train"]["label"] == list(range(10))
    assert loaded["test"]["label"] == list(reversed(range(10)))


def test_derive_class_names_falls_back_to_synthesized_names() -> None:
    train_split = Dataset.from_dict({"label": [0, 2, 1, 2]})

    names = eurosat.derive_class_names(train_split)

    assert names == ("class_0", "class_1", "class_2")


def test_derive_class_names_uses_classlabel_feature_names() -> None:
    features = Features({"label": ClassLabel(names=["Forest", "River", "SeaLake"])})
    train_split = Dataset.from_dict({"label": [0, 1, 2]}, features=features)

    names = eurosat.derive_class_names(train_split)

    assert names == ("Forest", "River", "SeaLake")


def test_eurosat_adapter_returns_rgb_tensor() -> None:
    records = Dataset.from_dict(
        {"image": [Image.new("RGB", (64, 64))], "label": [7]}
    )

    image, label = eurosat.EurosatDataset(records)[0]

    assert image.shape == (3, 64, 64)
    assert label == 7


def test_eurosat_adapter_augmented_output_still_valid_rgb_tensor() -> None:
    records = Dataset.from_dict(
        {"image": [Image.new("RGB", (64, 64))], "label": [3]}
    )

    image, label = eurosat.EurosatDataset(records, augment=True)[0]

    assert image.shape == (3, 64, 64)
    assert image.dtype == torch.float32
    assert label == 3


def test_eurosat_adapter_default_is_not_augmented() -> None:
    """EurosatDataset(records) with no augment kwarg must behave exactly like
    augment=False -- every existing positional-construction call site (e.g.
    server_loaders) relies on this default."""
    records = Dataset.from_dict(
        {"image": [Image.new("RGB", (64, 64))], "label": [1]}
    )

    plain_image, _ = eurosat.EurosatDataset(records)[0]
    explicit_image, _ = eurosat.EurosatDataset(records, augment=False)[0]

    assert torch.equal(plain_image, explicit_image)


def test_client_loaders_train_split_is_augmented_eval_split_is_not(
    monkeypatch,
) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "image": [Image.new("RGB", (64, 64)) for _ in range(20)],
                    "label": list(range(10)) * 2,
                }
            ),
            "test": Dataset.from_dict(
                {
                    "image": [Image.new("RGB", (64, 64)) for _ in range(4)],
                    "label": [0, 1, 2, 3],
                }
            ),
        }
    )
    monkeypatch.setattr(
        eurosat,
        "load_hf_dataset_cached",
        lambda _dataset_id, _cache_dir: dataset,
    )
    seen_augment_flags: list[bool] = []
    original_init = eurosat.EurosatDataset.__init__

    def spy_init(self, records, *, augment: bool = False) -> None:
        seen_augment_flags.append(augment)
        original_init(self, records, augment=augment)

    monkeypatch.setattr(eurosat.EurosatDataset, "__init__", spy_init)

    module = eurosat.EurosatDataModule()
    module.client_loaders(
        0, num_partitions=2, partition_mode="homogeneous", batch_size=4, seed=42
    )

    assert seen_augment_flags == [True, False]


def test_server_loaders_never_augmented(monkeypatch) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"image": [Image.new("RGB", (64, 64))], "label": [0]}
            ),
            "test": Dataset.from_dict(
                {
                    "image": [Image.new("RGB", (64, 64)) for _ in range(4)],
                    "label": [0, 1, 2, 3],
                }
            ),
        }
    )
    monkeypatch.setattr(
        eurosat,
        "load_hf_dataset_cached",
        lambda _dataset_id, _cache_dir: dataset,
    )
    seen_augment_flags: list[bool] = []
    original_init = eurosat.EurosatDataset.__init__

    def spy_init(self, records, *, augment: bool = False) -> None:
        seen_augment_flags.append(augment)
        original_init(self, records, augment=augment)

    monkeypatch.setattr(eurosat.EurosatDataset, "__init__", spy_init)

    module = eurosat.EurosatDataModule()
    module.server_loaders(batch_size=2, seed=42)

    assert seen_augment_flags == [False, False]


def test_data_module_class_names_uses_train_split_labels(monkeypatch) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"image": [Image.new("RGB", (64, 64))] * 3, "label": [0, 1, 2]}
            ),
            "test": Dataset.from_dict(
                {"image": [Image.new("RGB", (64, 64))] * 2, "label": [0, 1]}
            ),
        }
    )
    monkeypatch.setattr(
        eurosat,
        "load_hf_dataset_cached",
        lambda _dataset_id, _cache_dir: dataset,
    )

    module = eurosat.EurosatDataModule()

    assert module.class_names == ("class_0", "class_1", "class_2")
