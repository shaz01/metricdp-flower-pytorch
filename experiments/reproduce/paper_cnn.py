"""
## Model and training configuration:

The CNN, in Keras terms:
Conv2D(32, relu, input (128,128,1)) → MaxPool(2,2) → Conv2D(64, relu) → MaxPool(2,2)→
Conv2D(128, relu) → Flatten → Dense(64, relu) → Dropout(0.1) → Dense(32, relu) →
Dropout(0.1) → Dense(4, softmax).

Adam optimizer, sparse categorical cross-entropy, batch size 32, **5 local epochs per round, 20 rounds**.

---

Reproduction gotchas already visible: the kernel size of the Conv2D layers is *not stated* (3×3 is the overwhelming default, and their public code — Appendix H — would confirm), no pooling after the third conv is listed, and images presumably grayscale-normalized to 128×128 — check their repo for the exact preprocessing since the HF dataset ships 128×128 grayscale already.

"""

import torch
import torch.nn as nn


class PaperCNN(nn.Module):
    """The paper's four-class 128×128 grayscale CNN.

    TODO: The paper does not report convolution kernel sizes, so this reproduction
        uses the conventional 3×3 Keras ``Conv2D`` kernel with ``valid`` padding.
    """

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
        # With 3×3 valid convolutions, a 128×128 input becomes 128×28×28.
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 64),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 4),
            nn.Softmax(dim=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return four-class probabilities for grayscale 128×128 images."""
        return self.classifier(self.features(inputs))

