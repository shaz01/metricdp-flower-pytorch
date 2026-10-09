"""Small CNN used by the stacked-head experiment: a trainable body plus a 4-class linear head.

The private step only ever touches the head; the body is trained on the public examples alone and then frozen.
State-dict keys are ``body.*`` and ``head.weight`` / ``head.bias`` (the strategy rewrites the head by those names).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

FEATURES = 32
EPOCHS = (30, 100, 300)
LEARNING_RATES = (1e-3, 3e-3)
WEIGHT_DECAY = 1e-3


class Net(nn.Module):
    """Conv(1,8,3)-pool-Conv(8,16,3)-pool-Linear(400,32) feature body and a Linear(32,4) head."""

    def __init__(self) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(400, FEATURES),
            nn.ReLU(),
        )
        self.head = nn.Linear(FEATURES, 4)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(inputs))


def create_model() -> nn.Module:
    """Factory in the repo's ``model-module`` format (``experiments.stacked_head.cnn:create_model``)."""
    return Net()


def train_public(images: np.ndarray, labels: np.ndarray, epochs: int, learning_rate: float, seed: int) -> Net:
    """Full-batch Adam on the public set on CPU, so base models are reproducible across hosts."""
    torch.manual_seed(seed)
    net = Net()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    inputs, targets = torch.tensor(images), torch.tensor(labels)
    for _ in range(epochs):
        optimizer.zero_grad()
        nn.functional.cross_entropy(net(inputs), targets).backward()
        optimizer.step()
    return net.eval()


def embed(net: nn.Module, images: np.ndarray) -> np.ndarray:
    """Frozen-body features as float64 with an appended bias column, shape (n, FEATURES + 1)."""
    with torch.no_grad():
        features = net.body(torch.tensor(images)).numpy().astype(float)
    return np.column_stack((features, np.ones(len(features))))
