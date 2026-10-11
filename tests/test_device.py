"""Tests for metricdp_pytorch.utils.device.to_device."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from metricdp_pytorch.utils.device import to_device


def test_only_cuda_copies_are_asynchronous(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        torch.Tensor, "to",
        lambda self, device, non_blocking=False: calls.append((str(device), non_blocking)) or self,
    )
    for device in ("cpu", "mps", "cuda", "cuda:0"):
        to_device(torch.zeros(1), torch.device(device))

    assert calls == [("cpu", False), ("mps", False), ("cuda", True), ("cuda:0", True)]


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple MPS device")
def test_mps_training_with_to_device_is_reproducible() -> None:
    """Regression: asynchronous host-to-MPS copies made PaperCNN training irreproducible (NaN)."""
    from experiments.reproduce.paper_cnn import create_model
    from experiments.reproduce.paper_loss import sparse_categorical_cross_entropy

    generator = torch.Generator().manual_seed(0)
    data = TensorDataset(torch.rand(128, 1, 128, 128, generator=generator),
                         torch.randint(0, 4, (128,), generator=generator))
    device = torch.device("mps")

    def train() -> list[torch.Tensor]:
        torch.manual_seed(0)
        torch.mps.manual_seed(0)
        model = create_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(2):
            for images, labels in DataLoader(data, batch_size=32):
                optimizer.zero_grad()
                sparse_categorical_cross_entropy(
                    model(to_device(images, device)), to_device(labels, device)
                ).backward()
                optimizer.step()
        return [parameter.detach().cpu() for parameter in model.parameters()]

    first, second = train(), train()
    for left, right in zip(first, second, strict=True):
        assert torch.isfinite(left).all()
        assert torch.equal(left, right)
