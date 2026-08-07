"""Tests for the four-class CIFAR-10 data plugin."""

from __future__ import annotations

from datasets import Dataset, DatasetDict
from PIL import Image

from experiments.reproduce.dataset import cifar4


def test_load_cifar4_dataset_keeps_first_four_classes(monkeypatch) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"label": list(range(10))}),
            "test": Dataset.from_dict({"label": list(reversed(range(10)))}),
        }
    )
    monkeypatch.setattr(
        cifar4,
        "load_hf_dataset_cached",
        lambda _dataset_id, _cache_dir: dataset,
    )

    filtered = cifar4.load_cifar4_dataset()

    assert filtered["train"]["label"] == [0, 1, 2, 3]
    assert filtered["test"]["label"] == [3, 2, 1, 0]
    assert cifar4.CLASS_NAMES == ("airplane", "automobile", "bird", "cat")


def test_cifar4_adapter_returns_rgb_tensor() -> None:
    records = Dataset.from_dict(
        {"img": [Image.new("RGB", (32, 32))], "label": [2]}
    )

    image, label = cifar4.Cifar4Dataset(records)[0]

    assert image.shape == (3, 32, 32)
    assert label == 2
