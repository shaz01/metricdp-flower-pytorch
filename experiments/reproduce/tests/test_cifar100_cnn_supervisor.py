"""Tests for the supervisor-reference CIFAR-100 model."""

from __future__ import annotations

import torch
from torch import nn

from experiments.reproduce.cifar100_cnn_supervisor import Cifar100CNNSupervisor


def test_cifar100_cnn_supervisor_has_no_batchnorm() -> None:
    """BatchNorm has running-stats buffers that would poison the DP noise path
    (see the module docstring) -- GroupNorm was substituted throughout, and this
    guards that substitution stays in place.
    """
    model = Cifar100CNNSupervisor()

    for module in model.modules():
        assert not isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))


def test_cifar100_cnn_supervisor_has_no_buffers() -> None:
    """GroupNorm has no running-stats buffers -- unlike BatchNorm, this keeps the
    model's full state_dict() safe to transport through the DP noise path.
    """
    model = Cifar100CNNSupervisor()

    assert list(model.buffers()) == []


def test_cifar100_cnn_supervisor_forward_returns_probabilities() -> None:
    model = Cifar100CNNSupervisor()

    probabilities = model(torch.randn(2, 3, 32, 32))

    assert probabilities.shape == (2, 100)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-4)


def test_cifar100_cnn_supervisor_parameter_count_matches_design() -> None:
    model = Cifar100CNNSupervisor()

    param_count = sum(p.numel() for p in model.parameters())

    assert param_count == 4_631_268
