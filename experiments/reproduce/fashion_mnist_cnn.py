"""CNN model plugin for Fashion-MNIST reproduction runs."""

from __future__ import annotations

import torch
from torch import nn


class FashionMNISTCNN(nn.Module):
    """Four-class CNN for grayscale 28×28 Fashion-MNIST images."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
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
            nn.Linear(128 * 3 * 3, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 4),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return four-class probabilities for grayscale 28×28 images."""
        return self.classifier(self.features(inputs))


def create_model() -> FashionMNISTCNN:
    """Create the Fashion-MNIST model."""
    return FashionMNISTCNN()
