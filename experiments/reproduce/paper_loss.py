"""Loss function used by the paper's softmax-output CNN."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as functional
from flwr.app import ArrayRecord, MetricRecord
from torch.utils.data import DataLoader

from experiments.reproduce.paper_cnn import PaperCNN


def sparse_categorical_cross_entropy(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute sparse categorical cross-entropy from softmax probabilities. """

    tiny = torch.finfo(probabilities.dtype).tiny
    return functional.nll_loss(
        # Prevent log(0) by clamping to tiny.
        probabilities.clamp_min(tiny).log(),
        targets,
        reduction=reduction,
    )


def evaluate_model(
    model: torch.nn.Module,
    testloader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a softmax-output model with sparse CE and accuracy."""
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    examples = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            probabilities = model(images)
            total_loss += sparse_categorical_cross_entropy(
                probabilities, labels, reduction="sum"
            ).item()
            correct += (probabilities.argmax(dim=1) == labels).sum().item()
            examples += len(labels)
    if examples == 0:
        raise ValueError("The evaluation loader must not be empty.")
    return {"loss": total_loss / examples, "accuracy": correct / examples}


def make_evaluate_fn(
    testloader: DataLoader,
    device: torch.device | None = None,
) -> Callable[[int, ArrayRecord], MetricRecord]:
    """Create centralized sparse-CE loss/accuracy evaluation for ``PaperCNN``."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        del server_round
        model = PaperCNN()
        model.load_state_dict(arrays.to_torch_state_dict())
        return MetricRecord(evaluate_model(model, testloader, device))

    return evaluate

