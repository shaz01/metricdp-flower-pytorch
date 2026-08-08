"""Tests for the DenseNet+SELU CIFAR-100 model's building blocks."""

from __future__ import annotations

import math

import torch
from torch import nn

from experiments.reproduce.cifar100_cnn import (
    Cifar100CNN,
    DenseBlock,
    TransitionLayer,
    _lecun_normal_,
)


def test_dense_block_concatenates_growth_channels() -> None:
    block = DenseBlock(in_channels=64, num_layers=4, growth_rate=32)

    output = block(torch.randn(2, 64, 8, 8))

    assert block.out_channels == 64 + 4 * 32
    assert output.shape == (2, 192, 8, 8)


def test_transition_layer_compresses_and_downsamples() -> None:
    transition = TransitionLayer(in_channels=192, compression=0.5)

    output = transition(torch.randn(2, 192, 8, 8))

    assert transition.out_channels == 96
    assert output.shape == (2, 96, 4, 4)


def test_lecun_normal_init_matches_expected_std_for_conv() -> None:
    conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
    fan_in = 64 * 3 * 3

    _lecun_normal_(conv.weight)

    empirical_std = conv.weight.detach().std().item()
    assert math.isclose(empirical_std, math.sqrt(1.0 / fan_in), rel_tol=0.1)


def test_lecun_normal_init_matches_expected_std_for_linear() -> None:
    linear = nn.Linear(1000, 1000, bias=False)

    _lecun_normal_(linear.weight)

    empirical_std = linear.weight.detach().std().item()
    assert math.isclose(empirical_std, math.sqrt(1.0 / 1000), rel_tol=0.05)


def test_cifar100_cnn_channel_progression_matches_design() -> None:
    model = Cifar100CNN()

    assert model.block1.out_channels == 192
    assert model.transition1.out_channels == 96
    assert model.block2.out_channels == 224
    assert model.transition2.out_channels == 112
    assert model.block3.out_channels == 240


def test_cifar100_cnn_uses_alpha_dropout_not_regular_dropout() -> None:
    model = Cifar100CNN()

    assert isinstance(model.dropout, nn.AlphaDropout)
    assert not isinstance(model.dropout, nn.Dropout) or isinstance(
        model.dropout, nn.AlphaDropout
    )


def test_cifar100_cnn_forward_returns_probabilities() -> None:
    model = Cifar100CNN()

    probabilities = model(torch.randn(2, 3, 32, 32))

    assert probabilities.shape == (2, 100)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-5)
