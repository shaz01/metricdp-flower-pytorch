"""Tests for the paper's stateful-strategy model initialization."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import experiments.reproduce.paper_training as paper_training


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
    )

    assert losses == []
