"""Synthetic checks of the attack-side per-round shadow scalars (WP0 item 3)."""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.cia.cia import _calculate_loss, shadow_loss_and_gradient_norm


def _softmax_model() -> nn.Module:
    model = nn.Sequential(nn.Linear(2, 2, bias=False), nn.Softmax(dim=1))
    with torch.no_grad():
        model[0].weight.zero_()
    return model


def test_loss_and_gradient_norm_match_hand_computation() -> None:
    """Zero weights give p = (1/2, 1/2): loss ln 2 for every example.

    dL/dW = mean_i (p_i - y_i) x_i^T. With x1 = (1, 0), y1 = 0 and x2 = (0, 2),
    y2 = 1: ([[-.5, 0], [.5, 0]] + [[0, 1], [0, -1]]) / 2, Frobenius norm sqrt(0.625).
    Batch size 1 checks that gradients are accumulated over batches.
    """
    loader = DataLoader(
        TensorDataset(torch.tensor([[1.0, 0.0], [0.0, 2.0]]), torch.tensor([0, 1])),
        batch_size=1,
    )
    model = _softmax_model()

    loss, gradient_norm = shadow_loss_and_gradient_norm(model, loader, torch.device("cpu"))

    assert loss == pytest.approx(math.log(2.0))
    assert gradient_norm == pytest.approx(math.sqrt(0.625))
    assert loss == pytest.approx(_calculate_loss(model, loader, torch.device("cpu")))
    assert all(parameter.grad is None for parameter in model.parameters())


def test_gradient_norm_does_not_depend_on_batching() -> None:
    generator = torch.Generator().manual_seed(0)
    data = TensorDataset(torch.randn(10, 2, generator=generator), torch.randint(0, 2, (10,), generator=generator))
    model = _softmax_model()
    with torch.no_grad():
        model[0].weight.copy_(torch.tensor([[0.3, -0.2], [0.1, 0.4]]))

    single = shadow_loss_and_gradient_norm(model, DataLoader(data, batch_size=10), torch.device("cpu"))
    batched = shadow_loss_and_gradient_norm(model, DataLoader(data, batch_size=3), torch.device("cpu"))

    assert single == pytest.approx(batched)


def test_empty_loader_is_rejected() -> None:
    empty = DataLoader(TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long)))
    with pytest.raises(ValueError, match="empty"):
        shadow_loss_and_gradient_norm(_softmax_model(), empty, torch.device("cpu"))


def test_only_cuda_copies_are_asynchronous(monkeypatch) -> None:
    from experiments.cia import cia

    calls = []
    monkeypatch.setattr(torch.Tensor, "to", lambda self, device, non_blocking=False: calls.append(
        (str(device), non_blocking)) or self)
    for device in ("cpu", "mps", "cuda"):
        cia._to_device(torch.zeros(1), torch.device(device))

    assert calls == [("cpu", False), ("mps", False), ("cuda", True)]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple MPS device")
def test_mps_shadow_scalars_match_cpu() -> None:
    """Regression: asynchronous host-to-MPS copies corrupted evaluation batches."""
    generator = torch.Generator().manual_seed(0)
    data = TensorDataset(torch.rand(256, 3, 32, 32, generator=generator),
                         torch.randint(0, 4, (256,), generator=generator))
    torch.manual_seed(0)
    model = nn.Sequential(nn.Conv2d(3, 8, 3), nn.ReLU(), nn.Flatten(),
                          nn.Linear(8 * 30 * 30, 4), nn.Softmax(dim=1))
    loader = DataLoader(data, batch_size=32)

    cpu = shadow_loss_and_gradient_norm(model, loader, torch.device("cpu"))
    mps = [shadow_loss_and_gradient_norm(model, loader, torch.device("mps")) for _ in range(3)]
    plain = [_calculate_loss(model, loader, torch.device("mps")) for _ in range(3)]

    for loss, gradient_norm in mps:
        assert loss == pytest.approx(cpu[0], rel=1e-4)
        assert gradient_norm == pytest.approx(cpu[1], rel=1e-3)
    assert plain == pytest.approx([cpu[0]] * 3, rel=1e-4)
