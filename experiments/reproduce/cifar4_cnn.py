"""CNN model plugin for four-class CIFAR-10 runs."""

from __future__ import annotations

import torch
from torch import nn


class Cifar4CNN(nn.Module):
    """Four-class CNN for RGB 32×32 CIFAR-10 images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 4),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return four-class probabilities for RGB 32×32 images."""
        return self.classifier(self.features(inputs))


def create_model() -> Cifar4CNN:
    """Create the four-class CIFAR-10 model."""
    return Cifar4CNN()
