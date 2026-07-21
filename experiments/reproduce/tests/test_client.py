"""Tests for the paper reproduction Flower ClientApp."""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import experiments.reproduce.client as paper_client


class TinyPaperCNN(nn.Module):
    """Small softmax model used to test ClientApp message plumbing."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Sequential(nn.Linear(2, 4), nn.Softmax(dim=1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs)


def _loader() -> DataLoader:
    inputs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2, 3])
    return DataLoader(TensorDataset(inputs, labels), batch_size=2)


def _context() -> Context:
    return Context(
        run_id=1,
        node_id=1,
        node_config={"partition-id": 0, "num-partitions": 4},
        state=RecordDict(),
        run_config={
            "seed": 42,
            "num-clients": 4,
            "batch-size": 32,
            "partition-mode": "homogeneous",
            "local-epochs": 5,
            "data-module": "example.data:create_data_module",
            "partition-profile": "auto",
            "client-weights": "",
            "max-client-samples": 0,
        },
    )


def _request(message_type: str, *, proximal_mu: float = 0.0) -> Message:
    records = {"arrays": ArrayRecord(TinyPaperCNN().state_dict())}
    if message_type == "train":
        records["config"] = ConfigRecord(
            {"lr": 0.001, "proximal-mu": proximal_mu}
        )
    return Message(
        content=RecordDict(records),
        message_type=message_type,
        dst_node_id=1,
    )


def test_train_uses_paper_epochs_adam_config_and_fedprox(
    monkeypatch,
) -> None:
    calls: list[dict[str, float | int]] = []
    loader = _loader()

    def fake_train(
        model,
        trainloader,
        *,
        epochs,
        learning_rate,
        device,
        proximal_mu,
    ):
        del model, device
        assert trainloader is loader
        calls.append(
            {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "proximal_mu": proximal_mu,
            }
        )
        return [0.8, 0.6]

    monkeypatch.setattr(paper_client, "PaperCNN", TinyPaperCNN)
    monkeypatch.setattr(paper_client, "_client_data", lambda context: (loader, loader))
    monkeypatch.setattr(paper_client, "train_with_adam", fake_train)

    reply = paper_client.train(_request("train", proximal_mu=0.5), _context())

    assert calls == [
        {"epochs": 5, "learning_rate": 0.001, "proximal_mu": 0.5}
    ]
    assert reply.content["metrics"]["train_loss"] == 0.6
    assert reply.content["metrics"]["train_loss_mean"] == 0.7
    assert reply.content["metrics"]["num-examples"] == 4
    assert isinstance(reply.content["arrays"], ArrayRecord)


def test_evaluate_reports_local_loss_accuracy_and_count(monkeypatch) -> None:
    loader = _loader()
    monkeypatch.setattr(paper_client, "PaperCNN", TinyPaperCNN)
    monkeypatch.setattr(paper_client, "_client_data", lambda context: (loader, loader))
    monkeypatch.setattr(
        paper_client,
        "evaluate_model",
        lambda model, testloader, device: {"loss": 0.25, "accuracy": 0.75},
    )

    reply = paper_client.evaluate(_request("evaluate"), _context())

    assert reply.content["metrics"]["eval_loss"] == 0.25
    assert reply.content["metrics"]["eval_acc"] == 0.75
    assert reply.content["metrics"]["num-examples"] == 4


def test_client_data_is_loaded_through_configured_plugin(monkeypatch) -> None:
    loader = _loader()
    calls = []

    class FakeDataModule:
        def client_loaders(self, partition_id, **kwargs):
            calls.append((partition_id, kwargs))
            return loader, loader

    monkeypatch.setattr(
        paper_client,
        "load_data_module",
        lambda path, config: FakeDataModule(),
    )

    trainloader, testloader = paper_client._client_data(_context())

    assert trainloader is testloader is loader
    assert calls[0][0] == 0
    assert calls[0][1]["num_partitions"] == 4
    assert calls[0][1]["partition_mode"] == "homogeneous"


def test_partition_configuration_parsing() -> None:
    assert paper_client._client_weights({}) is None
    assert paper_client._client_weights({"client-weights": "1, 2, 3"}) == [
        1.0,
        2.0,
        3.0,
    ]
