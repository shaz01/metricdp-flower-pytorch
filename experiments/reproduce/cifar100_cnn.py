"""CNN model plugin for full 100-class CIFAR-100 runs.

v5: this repo's CIFAR-100 model, an adaptation of the project supervisor's
own `CNNCIFAR100` reference architecture (3 blocks of
2x[Conv3x3-BN-ReLU], channels 128/256/512, global-average-pooled
classifier). Replaced an earlier DenseNet+SELU architecture (v4, built and
verified for robustness to clipping-related weight-space update-magnitude
freezes found in this repo's original plain 3/4-block CNN line, v1-v3),
per project-owner direction consolidating this repo on a single CIFAR-100
model. See experiments/cifar100_scaling/sweep_cifar100_scaling.py's
docstring for the full model-history record.

Two changes from the supervisor's supplied source, both required for this
repo's DP mechanism and training pipeline, everything else preserved as
supplied:

1. `BatchNorm2d` -> `GroupNorm(min(32, num_features), num_features)`. This
   mirrors the exact group-count rule from the supplied file's own
   `_batchnorm_to_groupnorm` helper (`min(32, module.num_features)`,
   citing the Group Normalization paper's default of 32 groups). BatchNorm
   has running-stats buffers, which would poison this repo's DP noise path
   (weights transport as a full `state_dict()` into an `ArrayRecord`).
   The supplied file's own `replace_batchnorm_with_groupnorm` utility (a
   runtime reflection-based module-surgery pass) is not adopted here --
   this repo's model plugins define their architecture statically, and
   duplicating that utility for one call site adds a maintenance surface
   for no benefit.
2. Appended `Softmax(dim=1)`. The shared loss function
   (`experiments/reproduce/paper_loss.py`) consumes probabilities, not
   logits; the supplied source returns raw logits from `nn.Linear` with no
   final activation, matching every other model in the supplied file but
   not this repo's plugin contract.

`num_classes=100` and `p=0.5` (dropout rate) are hardcoded, not
constructor parameters -- the supplied source parameterizes both, but
every model plugin in this repo takes no constructor arguments, because
`metricdp_pytorch/model_module.py`'s `load_model()` calls the factory with
zero arguments.
"""

from __future__ import annotations

import torch
from torch import nn


class Cifar100CNN(nn.Module):
    """100-class CNN for RGB 32x32 CIFAR-100 images (adapted from the
    project supervisor's CNNCIFAR100 reference architecture -- see module
    docstring)."""

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
                nn.Dropout2d(0.5),
            )

        self.features = nn.Sequential(
            block(3, 128),
            block(128, 256),
            block(256, 512),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 100),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return 100-class probabilities for RGB 32x32 images."""
        return self.classifier(self.features(inputs))


def create_model() -> Cifar100CNN:
    """Create the 100-class CIFAR-100 model."""
    return Cifar100CNN()
