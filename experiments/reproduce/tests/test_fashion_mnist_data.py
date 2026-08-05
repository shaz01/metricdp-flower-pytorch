"""Tests for the four-class Fashion-MNIST data plugin."""

from __future__ import annotations

from datasets import Dataset, DatasetDict

from experiments.reproduce.dataset import fashion_mnist


def test_load_fashion_mnist_dataset_keeps_first_four_classes(monkeypatch) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"label": list(range(10))}),
            "test": Dataset.from_dict({"label": list(reversed(range(10)))}),
        }
    )
    monkeypatch.setattr(
        fashion_mnist,
        "load_hf_dataset_cached",
        lambda _dataset_id, _cache_dir: dataset,
    )

    filtered = fashion_mnist.load_fashion_mnist_dataset()

    assert filtered["train"]["label"] == [0, 1, 2, 3]
    assert filtered["test"]["label"] == [3, 2, 1, 0]
    assert fashion_mnist.CLASS_NAMES == (
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
    )
