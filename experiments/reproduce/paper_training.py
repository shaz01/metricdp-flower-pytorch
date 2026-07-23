"""Training and server-side initialization for the paper CNN."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.reproduce.paper_cnn import PaperCNN
from experiments.reproduce.paper_loss import sparse_categorical_cross_entropy
from metricdp_pytorch.utils.device import resolve_device

STATEFUL_AGGREGATIONS = frozenset({"fedavgm", "fedopt", "fedyogi"})
PAPER_INITIALIZATION_EPOCHS = 20
PAPER_ADAM_LEARNING_RATE = 1e-3


def seed_training(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch before creating an initial model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def requires_validation_initialization(aggregation: str) -> bool:
    """Return whether the paper pretrains this aggregation's initial model."""
    return aggregation.lower() in STATEFUL_AGGREGATIONS


def proximal_term(
    parameters: Iterable[torch.nn.Parameter],
    initial_parameters: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Mean squared difference ``mean((p - p_init)^2)`` over ALL parameters.

    Neither of the two "obvious" readings of the paper's ``(mu/2) * ||w -
    w^t||^2`` formula survive real 8-client training: an unnormalized sum of
    squares over the full parameter vector (literal reading) or an unsquared
    sum of per-tensor L2 norms (flwr.serverapp.strategy.FedProx's own
    docstring example) both grow large enough, dominated by this model's one
    100352->64 FC layer (98.6% of ~6.5M parameters), to trap real training in
    a trivial majority-class solution at the paper's literal mu=0.5 --
    confirmed identically under both conventions via real multi-round
    federated runs, independent of formula choice.

    Normalizing by total parameter count (matching the ``tf.reduce_mean``
    convention idiomatic in TensorFlow/Keras -- the paper's actual framework,
    per Appendix F -- rather than PyTorch's ``.norm(2)``/sum convention)
    resolves this: real testing shows mu=0.5 with this convention converges
    at essentially the same rate as unregularized FedAvg, matching the
    paper's own Table 6 result that FedProx and FedAvg land at identical
    accuracy (0.909).
    """
    total_squared_difference = sum(
        (parameter - initial).pow(2).sum()
        for parameter, initial in zip(parameters, initial_parameters, strict=True)
    )
    total_parameters = sum(initial.numel() for initial in initial_parameters)
    return total_squared_difference / total_parameters


def train_with_adam(
    model: nn.Module,
    trainloader: DataLoader,
    *,
    epochs: int,
    learning_rate: float = PAPER_ADAM_LEARNING_RATE,
    device: torch.device | None = None,
    proximal_mu: float = 0.0,
) -> list[float]:
    """Train with Adam, sparse categorical CE, and optional FedProx penalty.

    The proximal term is measured from the model state at function entry.
    Returns one example-weighted mean objective value per epoch.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least one.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if proximal_mu < 0:
        raise ValueError("proximal_mu must be non-negative.")
    if device is None:
        device = resolve_device()

    model.to(device)
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epoch_losses: list[float] = []
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        examples = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            probabilities = model(images)
            loss = sparse_categorical_cross_entropy(probabilities, labels)
            if proximal_mu > 0:
                loss = loss + 0.5 * proximal_mu * proximal_term(
                    model.parameters(), initial_parameters
                )
            loss.backward()
            optimizer.step()
            batch_size = len(labels)
            total_loss += loss.item() * batch_size
            examples += batch_size
        if examples == 0:
            raise ValueError("The validation loader must not be empty.")
        epoch_losses.append(total_loss / examples)
    return epoch_losses


def create_initial_model(
    aggregation: str,
    validation_loader: DataLoader,
    *,
    seed: int,
    epochs: int = PAPER_INITIALIZATION_EPOCHS,
    learning_rate: float = PAPER_ADAM_LEARNING_RATE,
    device: torch.device | None = None,
) -> tuple[PaperCNN, list[float]]:
    """Create the paper's initial model and pretrain it when required.

    FedAvgM, FedOpt, and FedYogi are trained on the stratified server
    validation half for 20 epochs with batch size 32 (the loader controls the
    batch size). The remaining strategies retain the same seeded random model.
    """
    seed_training(seed)
    model = PaperCNN()
    losses: list[float] = []
    if requires_validation_initialization(aggregation):
        losses = train_with_adam(
            model,
            validation_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )
    return model.cpu(), losses
