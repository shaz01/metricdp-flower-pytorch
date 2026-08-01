"""Tests for configurable model loading."""

from __future__ import annotations

import pytest
import torch

from metricdp_pytorch.model_module import load_model


def test_load_model_constructs_paper_model() -> None:
    model = load_model("experiments.reproduce.paper_cnn:create_model")

    probabilities = model(torch.randn(2, 1, 128, 128))

    assert probabilities.shape == (2, 4)


def test_fashion_mnist_model_matches_dataset_shape_and_classes() -> None:
    model = load_model("experiments.reproduce.fashion_mnist_cnn:create_model")

    probabilities = model(torch.randn(2, 1, 28, 28))

    assert probabilities.shape == (2, 10)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))


def test_load_model_rejects_non_model_factory() -> None:
    with pytest.raises(TypeError, match="did not return a torch.nn.Module"):
        load_model("experiments.reproduce.dataset.common:hf_offline")
