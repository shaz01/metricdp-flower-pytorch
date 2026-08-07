"""Tests for the full 100-class CIFAR-100 data plugin."""

from __future__ import annotations

import torch
from datasets import ClassLabel, Dataset, DatasetDict, Features
from PIL import Image

from experiments.reproduce.dataset import cifar100


def test_load_cifar100_dataset_keeps_all_classes(monkeypatch) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"fine_label": list(range(10))}),
            "test": Dataset.from_dict({"fine_label": list(reversed(range(10)))}),
        }
    )
    monkeypatch.setattr(
        cifar100,
        "load_hf_dataset_cached",
        lambda _dataset_id, _cache_dir: dataset,
    )

    loaded = cifar100.load_cifar100_dataset()

    assert loaded["train"]["fine_label"] == list(range(10))
    assert loaded["test"]["fine_label"] == list(reversed(range(10)))


def test_derive_class_names_falls_back_to_synthesized_names() -> None:
    train_split = Dataset.from_dict({"fine_label": [0, 2, 1, 2]})

    names = cifar100.derive_class_names(train_split)

    assert names == ("class_0", "class_1", "class_2")


def test_derive_class_names_uses_classlabel_feature_names() -> None:
    features = Features(
        {
            "fine_label": ClassLabel(names=["apple", "aquarium_fish", "baby"]),
        }
    )
    train_split = Dataset.from_dict({"fine_label": [0, 1, 2]}, features=features)

    names = cifar100.derive_class_names(train_split)

    assert names == ("apple", "aquarium_fish", "baby")


def test_cifar100_adapter_returns_rgb_tensor() -> None:
    records = Dataset.from_dict(
        {"img": [Image.new("RGB", (32, 32))], "fine_label": [42]}
    )

    image, label = cifar100.Cifar100Dataset(records)[0]

    assert image.shape == (3, 32, 32)
    assert label == 42


def test_cifar100_adapter_augmented_output_still_valid_rgb_tensor() -> None:
    records = Dataset.from_dict(
        {"img": [Image.new("RGB", (32, 32))], "fine_label": [7]}
    )

    image, label = cifar100.Cifar100Dataset(records, augment=True)[0]

    assert image.shape == (3, 32, 32)
    assert image.dtype == torch.float32
    assert label == 7


def test_cifar100_adapter_default_is_not_augmented() -> None:
    """Cifar100Dataset(records) with no augment kwarg must behave exactly
    like augment=False -- every existing positional-construction call site
    (e.g. server_loaders) relies on this default."""
    records = Dataset.from_dict(
        {"img": [Image.new("RGB", (32, 32))], "fine_label": [3]}
    )

    plain_image, _ = cifar100.Cifar100Dataset(records)[0]
    explicit_image, _ = cifar100.Cifar100Dataset(records, augment=False)[0]

    assert torch.equal(plain_image, explicit_image)


def test_client_loaders_train_split_is_augmented_eval_split_is_not(
    monkeypatch,
) -> None:
    """The two loaders returned by client_loaders must read through
    different Cifar100Dataset transforms -- augment=True for train,
    augment=False for eval -- verified by spying on the transform actually
    used rather than by pixel comparison (RandomCrop/RandomHorizontalFlip
    make exact equality checks flaky)."""
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "img": [Image.new("RGB", (32, 32)) for _ in range(20)],
                    "fine_label": list(range(20)),
                }
            ),
            "test": Dataset.from_dict(
                {
                    "img": [Image.new("RGB", (32, 32)) for _ in range(4)],
                    "fine_label": [0, 1, 2, 3],
                }
            ),
        }
    )
    monkeypatch.setattr(
        cifar100,
        "load_hf_dataset_cached",
        lambda _dataset_id, _cache_dir: dataset,
    )
    seen_augment_flags: list[bool] = []
    original_init = cifar100.Cifar100Dataset.__init__

    def spy_init(self, records, *, augment: bool = False) -> None:
        seen_augment_flags.append(augment)
        original_init(self, records, augment=augment)

    monkeypatch.setattr(cifar100.Cifar100Dataset, "__init__", spy_init)

    module = cifar100.Cifar100DataModule()
    module.client_loaders(
        0,
        num_partitions=2,
        partition_mode="homogeneous",
        batch_size=4,
        seed=42,
    )

    assert seen_augment_flags == [True, False]


def test_server_loaders_never_augmented(monkeypatch) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {"img": [Image.new("RGB", (32, 32))], "fine_label": [0]}
            ),
            "test": Dataset.from_dict(
                {
                    "img": [Image.new("RGB", (32, 32)) for _ in range(4)],
                    "fine_label": [0, 1, 2, 3],
                }
            ),
        }
    )
    monkeypatch.setattr(
        cifar100,
        "load_hf_dataset_cached",
        lambda _dataset_id, _cache_dir: dataset,
    )
    seen_augment_flags: list[bool] = []
    original_init = cifar100.Cifar100Dataset.__init__

    def spy_init(self, records, *, augment: bool = False) -> None:
        seen_augment_flags.append(augment)
        original_init(self, records, augment=augment)

    monkeypatch.setattr(cifar100.Cifar100Dataset, "__init__", spy_init)

    module = cifar100.Cifar100DataModule()
    module.server_loaders(batch_size=2, seed=42)

    assert seen_augment_flags == [False, False]


def test_data_module_class_names_uses_train_split_labels(monkeypatch) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "img": [Image.new("RGB", (32, 32))] * 3,
                    "fine_label": [0, 1, 2],
                }
            ),
            "test": Dataset.from_dict(
                {
                    "img": [Image.new("RGB", (32, 32))] * 2,
                    "fine_label": [0, 1],
                }
            ),
        }
    )
    monkeypatch.setattr(
        cifar100,
        "load_hf_dataset_cached",
        lambda _dataset_id, _cache_dir: dataset,
    )

    module = cifar100.Cifar100DataModule()

    assert module.class_names == ("class_0", "class_1", "class_2")
