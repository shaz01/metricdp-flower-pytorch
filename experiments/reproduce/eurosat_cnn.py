"""CNN model plugin for the 10-class EuroSAT accuracy sweep.

A plain 3-block Conv-GroupNorm-ReLU CNN, deliberately smaller than the CIFAR-100 model
(channels 32/64/128 vs. 128/256/512) -- EuroSAT is a much easier 10-way task on similarly-sized
64x64 images, so a lighter model trains faster without needing this repo's CIFAR-100 supervisor
reference architecture. Not tied to reproducing any external source, unlike cifar100_cnn.py.

GroupNorm (not BatchNorm) throughout -- BatchNorm has running-stats buffers, which would poison
this repo's DP noise path (weights transport as a full state_dict() into an ArrayRecord),
matching every other model plugin in this repo. Ends in Softmax(dim=1): the shared loss function
(experiments/reproduce/paper_loss.py) consumes probabilities, not logits.

num_classes=10 is hardcoded, not a constructor parameter -- every model plugin in this repo takes
no constructor arguments, because metricdp_pytorch/model_module.py's load_model() calls the
factory with zero arguments.

Verified parameter count: 289,194 (computed by direct execution during planning -- see
experiments/reproduce/tests/test_eurosat_cnn.py's test_eurosat_cnn_parameter_count_matches_design).
"""

from __future__ import annotations

import torch
from torch import nn


class EurosatCNN(nn.Module):
    """10-class CNN for RGB 64x64 EuroSAT images."""

    def __init__(self) -> None:
        super().__init__()

        def block(in_c: int, out_c: int) -> nn.Sequential:
            groups = min(32, out_c)
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.GroupNorm(groups, out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.GroupNorm(groups, out_c),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.3),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return 10-class probabilities for RGB 64x64 images."""
        return self.classifier(self.features(inputs))


def create_model() -> EurosatCNN:
    """Create the 10-class EuroSAT model."""
    return EurosatCNN()
