"""CNN model plugin for full 100-class CIFAR-100 runs."""

from __future__ import annotations

import torch
from torch import nn


class Cifar100CNN(nn.Module):
    """100-class CNN for RGB 32x32 CIFAR-100 images.

    v2 originally added a 4th conv block (512 channels) and GroupNorm after
    each conv. Live debugging found the 4th block itself -- not GroupNorm,
    not weight_decay, not noise_multiplier -- incompatible with weight-space
    clipping DP (metricdp_pytorch's and Flower's own
    DifferentialPrivacyServerSideFixedClipping-based strategies): the deeper
    model's natural per-round update magnitude (~29-30 in early rounds) far
    exceeds clipping_norm=5.0, and clipping it down permanently traps
    training at the random-chance floor for every privacy mode that clips
    (global-dp, metric-privacy) -- confirmed via a controlled comparison: the
    same clip/noise/weight-decay settings applied to this 3-block
    architecture converge cleanly (28.9% at n=8/r20/homogeneous/global-dp,
    matching this repo's own literature review of a documented FL failure
    mode: server-side update clipping colliding with a model's natural
    update scale). GroupNorm was independently ruled out (removing it alone,
    keeping 4 blocks, still froze) before isolating depth as the actual
    cause. See docs/superpowers/specs/2026-08-07-cifar100-v2-accuracy-design.md
    and this branch's debugging history for the full round-by-round
    evidence. Reverted to v1's 3-block depth as a result; augmentation and
    weight_decay (added elsewhere in this repo, not in this file) are kept.
    """

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
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
