"""Training and server-side initialization for the paper CNN."""

from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from experiments.reproduce.paper_loss import sparse_categorical_cross_entropy
from metricdp_pytorch.utils.device import resolve_device, to_device

STATEFUL_AGGREGATIONS = frozenset({"fedavgm", "fedopt", "fedyogi"})
PAPER_INITIALIZATION_EPOCHS = 20
PAPER_ADAM_LEARNING_RATE = 1e-3


def seed_training(seed: int) -> None:
    """Seed all supported RNGs and enable deterministic PyTorch kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
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


PROXIMAL_NORMALIZATIONS = ("mean", "sum")


def consensus_kl_term(
    local_probabilities: torch.Tensor,
    consensus_probabilities: torch.Tensor,
) -> torch.Tensor:
    """Batch mean of ``KL(p_consensus || p_local)`` from softmax probabilities.

    ``p_consensus`` comes from the global model a client received this round
    (a constant), so the term pulls the local model's predictions on its own
    inputs toward the consensus; it is zero when they agree.
    """
    tiny = torch.finfo(local_probabilities.dtype).tiny
    consensus = consensus_probabilities.detach()
    return (
        consensus
        * (consensus.clamp_min(tiny).log() - local_probabilities.clamp_min(tiny).log())
    ).sum(dim=1).mean()


def train_with_adam(
    model: nn.Module,
    trainloader: DataLoader,
    *,
    epochs: int,
    learning_rate: float = PAPER_ADAM_LEARNING_RATE,
    weight_decay: float = 0.0,
    device: torch.device | None = None,
    proximal_mu: float = 0.0,
    proximal_normalization: str = "mean",
    consensus_beta: float = 0.0,
    local_steps: int = 0,
) -> list[float]:
    """Train with Adam, sparse categorical CE, and optional FedProx penalty.

    The proximal term is measured from the model state at function entry:
    ``(mu/2) * proximal_term`` (``"mean"``, the repository convention) or
    ``(mu/2) * ||w - w_init||^2`` (``"sum"``, the FedProx paper's convention).
    ``consensus_beta > 0`` adds ``beta * consensus_kl_term`` against the model
    state at function entry (evaluated in eval mode, no gradient).
    ``local_steps > 0`` replaces the epoch count by a fixed number of optimizer
    steps, cycling through the loader as often as needed (the last pass may be
    partial). Returns one example-weighted mean objective value per pass.
    """
    if epochs < 1:
        raise ValueError("epochs must be at least one.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if proximal_mu < 0:
        raise ValueError("proximal_mu must be non-negative.")
    if proximal_normalization not in PROXIMAL_NORMALIZATIONS:
        raise ValueError(f"proximal_normalization must be one of {PROXIMAL_NORMALIZATIONS}.")
    if consensus_beta < 0:
        raise ValueError("consensus_beta must be non-negative.")
    if local_steps < 0:
        raise ValueError("local_steps must be non-negative.")
    if device is None:
        device = resolve_device()

    model.to(device)
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    # proximal_term is a mean; the "sum" convention multiplies it back by the count.
    proximal_scale = (
        sum(initial.numel() for initial in initial_parameters)
        if proximal_normalization == "sum"
        else 1
    )
    consensus_model = None
    if consensus_beta > 0:
        consensus_model = copy.deepcopy(model).eval()
        for parameter in consensus_model.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    epoch_losses: list[float] = []
    steps = 0
    while True:
        model.train()
        total_loss = 0.0
        examples = 0
        for images, labels in trainloader:
            images = to_device(images, device)
            labels = to_device(labels, device)
            optimizer.zero_grad()
            probabilities = model(images)
            loss = sparse_categorical_cross_entropy(probabilities, labels)
            if proximal_mu > 0:
                loss = loss + 0.5 * proximal_mu * proximal_scale * proximal_term(
                    model.parameters(), initial_parameters
                )
            if consensus_model is not None:
                with torch.no_grad():
                    consensus_probabilities = consensus_model(images)
                loss = loss + consensus_beta * consensus_kl_term(
                    probabilities, consensus_probabilities
                )
            loss.backward()
            optimizer.step()
            batch_size = len(labels)
            total_loss += loss.item() * batch_size
            examples += batch_size
            steps += 1
            if local_steps and steps == local_steps:
                break
        if examples == 0:
            raise ValueError("The validation loader must not be empty.")
        epoch_losses.append(total_loss / examples)
        if (steps == local_steps) if local_steps else (len(epoch_losses) == epochs):
            return epoch_losses


def cosine_decayed_lr(
    base_lr: float,
    server_round: int,
    num_rounds: int,
    *,
    min_fraction: float = 0.01,
) -> float:
    """Cosine-decay ``base_lr`` from round 1 to ``num_rounds``, floored at
    ``min_fraction * base_lr`` rather than zero (a literal zero LR at the
    final round would freeze training entirely for that round).

    ``num_rounds <= 1`` returns ``base_lr`` unchanged -- there's no
    trajectory to decay over a single-round run.
    """
    if num_rounds <= 1:
        return base_lr
    progress = min(max((server_round - 1) / (num_rounds - 1), 0.0), 1.0)
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_fraction + (1.0 - min_fraction) * decay)


def create_initial_model(
    aggregation: str,
    validation_loader: DataLoader,
    *,
    seed: int,
    epochs: int = PAPER_INITIALIZATION_EPOCHS,
    learning_rate: float = PAPER_ADAM_LEARNING_RATE,
    device: torch.device | None = None,
    model_factory: Callable[[], nn.Module],
) -> tuple[nn.Module, list[float]]:
    """Create the paper's initial model and pretrain it when required.

    FedAvgM, FedOpt, and FedYogi are trained on the stratified server
    validation half for 20 epochs with batch size 32 (the loader controls the
    batch size). The remaining strategies retain the same seeded random model.
    """
    seed_training(seed)
    selected_model = model_factory()
    losses: list[float] = []
    if requires_validation_initialization(aggregation):
        losses = train_with_adam(
            selected_model,
            validation_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            device=device,
        )
    return selected_model.cpu(), losses
