"""Training and server-side initialization for the paper CNN."""

from __future__ import annotations

import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.reproduce.paper_cnn import PaperCNN
from experiments.reproduce.paper_loss import sparse_categorical_cross_entropy

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
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
                proximal_penalty = sum(
                    torch.sum((parameter - initial) ** 2)
                    for parameter, initial in zip(
                        model.parameters(), initial_parameters, strict=True
                    )
                )
                loss = loss + 0.5 * proximal_mu * proximal_penalty
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
