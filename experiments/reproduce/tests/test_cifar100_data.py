"""Tests for the full 100-class CIFAR-100 data plugin."""

from __future__ import annotations

from datasets import Dataset, DatasetDict
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


def test_cifar100_adapter_returns_rgb_tensor() -> None:
    records = Dataset.from_dict(
        {"img": [Image.new("RGB", (32, 32))], "fine_label": [42]}
    )

    image, label = cifar100.Cifar100Dataset(records)[0]

    assert image.shape == (3, 32, 32)
    assert label == 42


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
