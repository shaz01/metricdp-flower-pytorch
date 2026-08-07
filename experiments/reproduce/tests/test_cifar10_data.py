"""Tests for the full ten-class CIFAR-10 data plugin."""

from __future__ import annotations

import torch
from datasets import Dataset, DatasetDict
from PIL import Image

from experiments.reproduce.dataset import cifar10
from metricdp_pytorch.model_module import load_model


def test_load_cifar10_dataset_keeps_all_ten_classes(monkeypatch) -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"label": list(range(10))}),
            "test": Dataset.from_dict({"label": list(reversed(range(10)))}),
        }
    )
    monkeypatch.setattr(
        cifar10,
        "load_hf_dataset_cached",
        lambda _dataset_id, _cache_dir: dataset,
    )

    loaded = cifar10.load_cifar10_dataset()

    assert loaded["train"]["label"] == list(range(10))
    assert loaded["test"]["label"] == list(reversed(range(10)))
    assert len(cifar10.CLASS_NAMES) == 10
    assert cifar10.CLASS_NAMES[0] == "airplane"
    assert cifar10.CLASS_NAMES[-1] == "truck"


def test_cifar10_adapter_returns_rgb_tensor() -> None:
    records = Dataset.from_dict(
        {"img": [Image.new("RGB", (32, 32))], "label": [7]}
    )

    image, label = cifar10.Cifar10Dataset(records)[0]

    assert image.shape == (3, 32, 32)
    assert label == 7


def test_cifar10_data_module_matches_model_output_size(monkeypatch) -> None:
    monkeypatch.setattr(
        cifar10,
        "load_hf_dataset_cached",
        lambda _dataset_id, _cache_dir: DatasetDict(
            {
                "train": Dataset.from_dict({"label": list(range(10))}),
                "test": Dataset.from_dict({"label": list(range(10))}),
            }
        ),
    )
    module = cifar10.create_data_module({})
    model = load_model("experiments.reproduce.cifar10_cnn:create_model")

    probabilities = model(torch.randn(2, 3, 32, 32))

    assert len(module.class_names) == probabilities.shape[1] == 10
