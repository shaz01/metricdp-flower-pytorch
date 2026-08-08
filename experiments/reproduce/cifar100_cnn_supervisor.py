"""CNN model plugin for full 100-class CIFAR-100 runs (supervisor reference).

This is the project supervisor's own `CNNCIFAR100` reference architecture
(wider than the CIFAR-10 model it sits alongside in the supplied file: 3
blocks of 2x[Conv3x3-BN-ReLU], channels 128/256/512, global-average-pooled
classifier), run as a second, independent CIFAR-100 model/sweep alongside
-- not instead of -- the DenseNet+SELU model in `cifar100_cnn.py` on the
sibling `feature/cifar100-scaling` branch.

Two changes from the supplied source, both required for this repo's DP
mechanism and training pipeline, everything else preserved as supplied:

1. `BatchNorm2d` -> `GroupNorm(min(32, num_features), num_features)`. This
   mirrors the exact group-count rule from the supplied file's own
   `_batchnorm_to_groupnorm` helper (`min(32, module.num_features)`,
   citing the Group Normalization paper's default of 32 groups) rather
   than the fixed `GroupNorm(8, ...)` used in this repo's other CIFAR-100
   model -- both are valid; this one follows the supervisor's own stated
   convention. BatchNorm has running-stats buffers, which would poison
   this repo's DP noise path (weights transport as a full `state_dict()`
   into an `ArrayRecord`; see `cifar100_cnn.py`'s docstring for the same
   reasoning applied to the sibling model). The supplied file's own
   `replace_batchnorm_with_groupnorm` utility (a runtime reflection-based
   module-surgery pass) is not adopted here -- this repo's other model
   plugins define their architecture statically, and duplicating that
   utility for one call site adds a maintenance surface for no benefit.
2. Appended `Softmax(dim=1)`. The shared loss function
   (`experiments/reproduce/paper_loss.py`) consumes probabilities, not
   logits; the supplied source returns raw logits from `nn.Linear` with no
   final activation, matching every other model in this file's source but
   not this repo's plugin contract.

`num_classes=100` and `p=0.5` (dropout rate) are hardcoded, not
constructor parameters -- the supplied source parameterizes both, but
every model plugin in this repo takes no constructor arguments, because
`metricdp_pytorch/model_module.py`'s `load_model()` calls the factory with
zero arguments.

See docs/superpowers/specs/2026-08-08-cifar100-supervisor-cnn-design.md
for the (now-superseded, see branch history) CIFAR-10-adaptation record;
this file implements the supervisor's actual CIFAR-100 architecture
supplied afterward, not that adaptation.
"""

from __future__ import annotations

import torch
from torch import nn


class Cifar100CNNSupervisor(nn.Module):
    """100-class CNN for RGB 32x32 CIFAR-100 images (supervisor's own
    CNNCIFAR100 reference architecture, adapted for this repo's DP
    mechanism and loss contract -- see module docstring)."""

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


def create_model() -> Cifar100CNNSupervisor:
    """Create the 100-class CIFAR-100 model."""
    return Cifar100CNNSupervisor()
