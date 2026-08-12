"""Tests for the EuroSAT model."""

from __future__ import annotations

import torch
from torch import nn

from experiments.reproduce.eurosat_cnn import EurosatCNN


def test_eurosat_cnn_has_no_batchnorm() -> None:
    """BatchNorm has running-stats buffers that would poison the DP noise path
    (weights transport as a full state_dict() into an ArrayRecord) -- GroupNorm
    is used throughout instead, and this guards that stays in place.
    """
    model = EurosatCNN()

    for module in model.modules():
        assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))


def test_eurosat_cnn_has_no_buffers() -> None:
    """GroupNorm has no running-stats buffers -- unlike BatchNorm, this keeps the
    model's full state_dict() safe to transport through the DP noise path.
    """
    model = EurosatCNN()

    assert list(model.buffers()) == []


def test_eurosat_cnn_forward_returns_probabilities() -> None:
    model = EurosatCNN()

    probabilities = model(torch.randn(2, 3, 64, 64))

    assert probabilities.shape == (2, 10)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-4)


def test_eurosat_cnn_parameter_count_matches_design() -> None:
    model = EurosatCNN()

    param_count = sum(p.numel() for p in model.parameters())

    assert param_count == 289_194
