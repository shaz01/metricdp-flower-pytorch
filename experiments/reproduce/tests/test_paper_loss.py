"""Tests for divergence-safe centralized evaluation."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.reproduce.paper_loss import evaluate_model
from metricdp_pytorch.utils.device import resolve_device


class ConstantOutputModel(nn.Module):
    """Ignore the input and always return a fixed probability tensor."""

    def __init__(self, output: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("output", output)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.output.expand(inputs.shape[0], -1)


def _loader() -> DataLoader:
    # 4 classes to match PaperCNN's real (Alzheimer, 4-class) usage --
    # label_binarize collapses to a single column for 2-class input, which
    # would raise an unrelated shape mismatch against probabilities.ravel().
    inputs = torch.zeros((4, 2), dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 3])
    return DataLoader(TensorDataset(inputs, labels), batch_size=2)


def test_evaluate_model_reports_finite_metrics_on_convergence() -> None:
    """A well-behaved model returns real sklearn-derived metrics, unflagged."""
    output = torch.tensor([0.4, 0.3, 0.2, 0.1])
    model = ConstantOutputModel(output)

    metrics = evaluate_model(model, _loader(), resolve_device())

    assert metrics["diverged"] == 0.0
    assert metrics["loss"] == metrics["loss"]  # not NaN


def test_evaluate_model_flags_divergence_instead_of_raising() -> None:
    """NaN probabilities (e.g. from too much DP noise) must not crash sklearn.

    Previously this let NaN reach sklearn's roc_curve/f1_score, which raises
    "Input contains NaN" and aborts the whole federated run -- losing every
    prior round's history (see Phase 1 of docs/RESEARCH_ROADMAP.md; 11/24
    high-noise-multiplier runs failed exactly this way with zero artifacts
    saved). It must instead report a finite, clearly-flagged result.
    """
    output = torch.tensor([float("nan"), float("nan"), float("nan"), float("nan")])
    model = ConstantOutputModel(output)

    metrics = evaluate_model(model, _loader(), resolve_device(), server_round=7)

    assert metrics["diverged"] == 1.0
    assert math.isfinite(metrics["loss"])  # a sentinel, never NaN/inf
    assert metrics["accuracy"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["precision"] == 0.0
    assert metrics["auc"] == 0.5
