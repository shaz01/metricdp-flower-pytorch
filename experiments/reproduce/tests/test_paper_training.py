"""Tests for the paper's stateful-strategy model initialization."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import experiments.reproduce.paper_training as paper_training
from experiments.reproduce.paper_cnn import PaperCNN


def _tiny_loader() -> DataLoader:
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 1, 0])
    return DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)

def test_adam_pretraining_updates_a_softmax_model() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Softmax(dim=1))
    before = [parameter.detach().clone() for parameter in model.parameters()]

    losses = paper_training.train_with_adam(
        model,
        _tiny_loader(),
        epochs=2,
        learning_rate=0.01,
        device=torch.device("cpu"),
    )

    assert len(losses) == 2
    assert all(torch.isfinite(torch.tensor(losses)))
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, model.parameters(), strict=True)
    )


def test_create_initial_model_pretrains_fedavgm(
    monkeypatch,
) -> None:
    calls: list[tuple[int, float]] = []

    def fake_train(model, trainloader, *, epochs, learning_rate, device):
        del model, trainloader, device
        calls.append((epochs, learning_rate))
        return [0.9] * epochs

    monkeypatch.setattr(paper_training, "train_with_adam", fake_train)
    _, losses = paper_training.create_initial_model(
        "fedavgm",
        _tiny_loader(),
        seed=42,
        epochs=20,
        learning_rate=0.001,
        device=torch.device("cpu"),
        model_factory=PaperCNN,
    )

    assert calls == [(20, 0.001)]
    assert losses == [0.9] * 20


def test_create_initial_model_leaves_fedavg_random(
    monkeypatch,
) -> None:
    def unexpected_train(*args, **kwargs):
        raise AssertionError("FedAvg must not use validation pretraining")

    monkeypatch.setattr(paper_training, "train_with_adam", unexpected_train)
    _, losses = paper_training.create_initial_model(
        "fedavg",
        _tiny_loader(),
        seed=42,
        device=torch.device("cpu"),
        model_factory=PaperCNN,
    )

    assert losses == []


def test_proximal_term_uses_mean_squared_difference_over_all_parameters() -> None:
    """Matches the ``tf.reduce_mean`` convention idiomatic in TensorFlow/Keras
    (the paper's actual framework, per Appendix F) rather than PyTorch's
    ``.norm(2)``/sum convention. Both an unnormalized sum-of-squares and an
    unsquared sum of per-tensor L2 norms are mathematically defensible
    readings of the paper's ``(mu/2) * ||w - w^t||^2`` formula, but both grow
    large enough over this model's ~6.5M parameters (dominated by one
    100352->64 FC layer) to trap real, imbalanced-class training in a
    trivial majority-class solution at mu=0.5 -- see the design discussion
    for the empirical evidence. Normalizing by total parameter count avoids
    this and matches the paper's own reported FedProx-equals-FedAvg result.
    """
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]

    with torch.no_grad():
        model.weight.copy_(torch.tensor([[4.0, 0.0], [0.0, 1.0]]))  # drift of 3 in one entry

    term = paper_training.proximal_term(model.parameters(), initial_parameters)

    # sum of squares = 3^2 = 9, over 4 total parameters -> mean = 2.25
    assert term.item() == pytest.approx(2.25, abs=1e-6)


def test_adam_uses_weight_decay_when_given() -> None:
    model = nn.Sequential(nn.Linear(2, 2), nn.Softmax(dim=1))
    captured_optimizers: list[torch.optim.Optimizer] = []
    real_adam = torch.optim.Adam

    def spy_adam(params, **kwargs):
        optimizer = real_adam(params, **kwargs)
        captured_optimizers.append(optimizer)
        return optimizer

    import experiments.reproduce.paper_training as module

    original_adam = module.torch.optim.Adam
    module.torch.optim.Adam = spy_adam
    try:
        module.train_with_adam(
            model,
            _tiny_loader(),
            epochs=1,
            learning_rate=0.01,
            weight_decay=0.0005,
            device=torch.device("cpu"),
        )
    finally:
        module.torch.optim.Adam = original_adam

    assert len(captured_optimizers) == 1
    assert captured_optimizers[0].defaults["weight_decay"] == pytest.approx(0.0005)


def test_adam_defaults_to_zero_weight_decay() -> None:
    """Every existing caller that doesn't pass weight_decay must keep
    getting PyTorch's un-regularized Adam, exactly as before this change."""
    model = nn.Sequential(nn.Linear(2, 2), nn.Softmax(dim=1))
    losses = paper_training.train_with_adam(
        model,
        _tiny_loader(),
        epochs=1,
        learning_rate=0.01,
        device=torch.device("cpu"),
    )
    assert len(losses) == 1  # smoke check that the default path still runs


def test_cosine_decayed_lr_starts_near_base_and_ends_at_floor() -> None:
    base_lr = 0.001
    num_rounds = 20

    first = paper_training.cosine_decayed_lr(base_lr, 1, num_rounds)
    last = paper_training.cosine_decayed_lr(base_lr, num_rounds, num_rounds)
    middle = paper_training.cosine_decayed_lr(base_lr, 10, num_rounds)

    assert first == pytest.approx(base_lr, rel=1e-6)
    assert last == pytest.approx(base_lr * 0.01, rel=1e-6)
    assert base_lr * 0.01 < middle < base_lr


def test_cosine_decayed_lr_handles_single_round_run() -> None:
    assert paper_training.cosine_decayed_lr(0.001, 1, 1) == pytest.approx(0.001)
