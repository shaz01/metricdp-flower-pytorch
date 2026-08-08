"""CNN model plugin for full 100-class CIFAR-100 runs."""

from __future__ import annotations

import math

import torch
from torch import nn


def _lecun_normal_(weight: torch.Tensor) -> None:
    """Initialize ``weight`` in place with LeCun-normal std, required for SELU.

    fan_in is the number of inputs each output element is computed from:
    in_channels * kernel_h * kernel_w for a Conv2d weight (shape
    out_channels x in_channels x kh x kw), or in_features for a Linear
    weight (shape out_features x in_features). Computed manually rather
    than via torch.nn.init's private fan-calculation helper.
    """
    if weight.dim() == 4:
        fan_in = weight.shape[1] * weight.shape[2] * weight.shape[3]
    elif weight.dim() == 2:
        fan_in = weight.shape[1]
    else:
        raise ValueError(
            f"Unsupported weight shape for LeCun init: {tuple(weight.shape)}"
        )
    std = math.sqrt(1.0 / fan_in)
    nn.init.normal_(weight, mean=0.0, std=std)


class _DenseLayer(nn.Module):
    """One dense-block layer: GroupNorm -> SELU -> Conv2d (pre-activation).

    No 1x1 bottleneck conv -- this network is small enough that the
    bottleneck's compute savings aren't needed, and it's one fewer
    untested moving part (DenseNet-BC convention, simplified).
    """

    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(8, in_channels)
        self.activation = nn.SELU()
        self.conv = nn.Conv2d(
            in_channels, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(self.activation(self.norm(inputs)))


class DenseBlock(nn.Module):
    """``num_layers`` dense layers, each concatenated onto a growing feature
    map (not summed) -- DenseNet's concatenative skip-connection convention,
    chosen (over additive ResNet-style residuals) per project direction.
    """

    def __init__(self, in_channels: int, num_layers: int, growth_rate: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _DenseLayer(in_channels + i * growth_rate, growth_rate)
                for i in range(num_layers)
            ]
        )
        self.out_channels = in_channels + num_layers * growth_rate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = inputs
        for layer in self.layers:
            features = torch.cat([features, layer(features)], dim=1)
        return features


class TransitionLayer(nn.Module):
    """GroupNorm -> SELU -> 1x1 Conv (channel compression) -> 2x2 AvgPool."""

    def __init__(self, in_channels: int, compression: float = 0.5) -> None:
        super().__init__()
        self.out_channels = max(1, int(in_channels * compression))
        self.norm = nn.GroupNorm(8, in_channels)
        self.activation = nn.SELU()
        self.conv = nn.Conv2d(
            in_channels, self.out_channels, kernel_size=1, bias=False
        )
        self.pool = nn.AvgPool2d(2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(self.activation(self.norm(inputs))))


class Cifar100CNN(nn.Module):
    """100-class DenseNet+SELU CNN for RGB 32x32 CIFAR-100 images.

    v4: full architecture replacement, not a variant of v1/v2/v3's plain
    CNN. Two separate anomalies in v3's 3-block model motivated this: (1) a
    4th conv block's natural per-round update magnitude (~29-30) exceeded
    clipping_norm=5.0 and froze every clipping privacy mode (see
    docs/superpowers/specs/2026-08-07-cifar100-v2-accuracy-design.md); (2)
    even the reverted 3-block model froze *vanilla* specifically at
    n=128/homogeneous -- 128 highly-correlated client updates reinforced
    rather than averaged out (no 1/sqrt(n) reduction), producing a combined
    per-round step too large for the model to absorb on round 1. Both point
    to weight-space update-magnitude sensitivity as an architecture
    property, not just a hyperparameter to retune. Replaced with a
    SmoothNets-inspired (arXiv 2205.04095) design: concatenative
    (DenseNet-style) skip connections, GroupNorm(8), SELU activation with
    LeCun-normal init and AlphaDropout, and width over depth (3 dense
    blocks x 4 layers, growth_rate=32 -- deliberately shallow given the
    depth finding above). See
    docs/superpowers/specs/2026-08-08-cifar100-densenet-selu-design.md for
    the full design and verification record.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.block1 = DenseBlock(64, num_layers=4, growth_rate=32)
        self.transition1 = TransitionLayer(self.block1.out_channels)
        self.block2 = DenseBlock(
            self.transition1.out_channels, num_layers=4, growth_rate=32
        )
        self.transition2 = TransitionLayer(self.block2.out_channels)
        self.block3 = DenseBlock(
            self.transition2.out_channels, num_layers=4, growth_rate=32
        )
        self.final_norm = nn.GroupNorm(8, self.block3.out_channels)
        self.final_activation = nn.SELU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.AlphaDropout(p=0.1)
        self.classifier = nn.Linear(self.block3.out_channels, 100)
        self.softmax = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                _lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return 100-class probabilities for RGB 32x32 images."""
        features = self.stem(inputs)
        features = self.block1(features)
        features = self.transition1(features)
        features = self.block2(features)
        features = self.transition2(features)
        features = self.block3(features)
        features = self.final_activation(self.final_norm(features))
        features = self.pool(features).flatten(1)
        features = self.dropout(features)
        return self.softmax(self.classifier(features))


def create_model() -> Cifar100CNN:
    """Create the 100-class CIFAR-100 model."""
    return Cifar100CNN()
