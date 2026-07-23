"""Loss function used by the paper's softmax-output CNN."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as functional
from flwr.app import ArrayRecord, MetricRecord
from sklearn.metrics import auc, f1_score, precision_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from experiments.reproduce.paper_cnn import PaperCNN
from metricdp_pytorch.utils.device import resolve_device


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
    """Evaluate a softmax-output model: loss, accuracy, weighted F1/precision
    (Tables 6/8), and micro-averaged one-vs-rest AUC (Appendix C)."""
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    examples = 0
    all_labels: list[torch.Tensor] = []
    all_probabilities: list[torch.Tensor] = []
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            probabilities = model(images)
            total_loss += sparse_categorical_cross_entropy(
                probabilities, labels, reduction="sum"
            ).item()
            correct += (probabilities.argmax(dim=1) == labels).sum().item()
            examples += len(labels)
            all_labels.append(labels.cpu())
            all_probabilities.append(probabilities.cpu())
    if examples == 0:
        raise ValueError("The evaluation loader must not be empty.")

    labels_array = torch.cat(all_labels).numpy()
    probabilities_array = torch.cat(all_probabilities).numpy()
    predictions_array = probabilities_array.argmax(axis=1)
    num_classes = probabilities_array.shape[1]

    f1 = f1_score(labels_array, predictions_array, average="weighted", zero_division=0)
    precision = precision_score(
        labels_array, predictions_array, average="weighted", zero_division=0
    )

    # Micro-averaged one-vs-rest AUC: binarize labels against all classes,
    # then ravel true/score matrices before the ROC curve (Appendix C: "the
    # micro-averaged one-vs-rest ROC-AUC score has been used, comparing each
    # class against the other three").
    true_binarized = label_binarize(labels_array, classes=list(range(num_classes)))
    false_positive_rate, true_positive_rate, _ = roc_curve(
        true_binarized.ravel(), probabilities_array.ravel()
    )
    auc_micro = auc(false_positive_rate, true_positive_rate)

    return {
        "loss": total_loss / examples,
        "accuracy": correct / examples,
        "f1": float(f1),
        "precision": float(precision),
        "auc": float(auc_micro),
    }


def make_evaluate_fn(
    testloader: DataLoader,
    device: torch.device | None = None,
) -> Callable[[int, ArrayRecord], MetricRecord]:
    """Create centralized sparse-CE loss/accuracy evaluation for ``PaperCNN``."""
    if device is None:
        device = resolve_device()

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        del server_round
        model = PaperCNN()
        model.load_state_dict(arrays.to_torch_state_dict())
        return MetricRecord(evaluate_model(model, testloader, device))

    return evaluate
