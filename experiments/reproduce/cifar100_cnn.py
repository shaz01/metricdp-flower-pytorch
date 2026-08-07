"""CNN model plugin for full 100-class CIFAR-100 runs."""

from __future__ import annotations

import torch
from torch import nn


class Cifar100CNN(nn.Module):
    """100-class CNN for RGB 32x32 CIFAR-100 images.

    v2: adds a 4th conv block and GroupNorm after each conv (v1 had 3 blocks,
    no normalization). GroupNorm has no running-stats buffers -- unlike
    BatchNorm2d -- so it needs no special-casing in the metric-privacy
    pairwise-distance/aggregation code (metricdp_pytorch/metricdp_strategy.py),
    which was the reason v1 excluded normalization entirely. The extra
    pooling stage from the 4th block shrinks the flatten dimension
    (4096->2048), so total parameter count barely grows (~2.6M -> ~2.76M)
    despite the added depth.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.GroupNorm(8, 512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 100),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return 100-class probabilities for RGB 32×32 images."""
        return self.classifier(self.features(inputs))


def create_model() -> Cifar100CNN:
    """Create the 100-class CIFAR-100 model."""
    return Cifar100CNN()
