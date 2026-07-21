"""Flower ClientApp for the Alzheimer MRI paper reproduction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from experiments.reproduce.paper_cnn import PaperCNN
from experiments.reproduce.paper_loss import evaluate_model
from experiments.reproduce.paper_training import seed_training, train_with_adam
from metricdp_pytorch.data_module import load_data_module
from metricdp_pytorch.utils.runtime import runtime_config

app = ClientApp()



def _client_weights(config: Mapping[str, Any]) -> Sequence[float] | None:
    """Return optional non-IID quantity weights from run configuration."""
    weights = config.get("client-weights", "")
    if not weights:
        return None
    if isinstance(weights, str):
        return [float(weight.strip()) for weight in weights.split(",") if weight.strip()]
    if not isinstance(weights, (list, tuple)):
        raise TypeError("client-weights must be a list or comma-separated string.")
    return [float(weight) for weight in weights]


def _client_data(context: Context):
    """Load this SuperNode's deterministic local train and test loaders."""
    config = runtime_config(context)
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(config["num-clients"])
    if partition_id >= num_partitions:
        raise ValueError(
            f"Partition {partition_id} is outside the configured "
            f"{num_partitions} data clients. Match num-supernodes to num-clients."
        )
    data_module = load_data_module(str(config["data-module"]), config)
    return data_module.client_loaders(
        partition_id,
        batch_size=int(config["batch-size"]),
        partition_mode=str(config["partition-mode"]),
        num_partitions=num_partitions,
        seed=int(config.get("seed", 42)),
        partition_profile=str(config.get("partition-profile", "auto")),
        client_weights=_client_weights(config),
        max_samples=int(config.get("max-client-samples", 0)),
    )


@app.train()
def train(msg: Message, context: Context) -> Message:
    """Train the received global PaperCNN on one MRI client partition."""
    run_config = runtime_config(context)
    partition_id = int(context.node_config["partition-id"])
    seed_training(int(run_config.get("seed", 42)) + partition_id)

    model = PaperCNN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    trainloader, _ = _client_data(context)
    train_config = msg.content["config"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_losses = train_with_adam(
        model,
        trainloader,
        epochs=int(run_config.get("local-epochs", 5)),
        learning_rate=float(train_config["lr"]),
        device=device,
        proximal_mu=float(train_config.get("proximal-mu", 0.0)),
    )
    model.cpu()

    return Message(
        content=RecordDict(
            {
                "arrays": ArrayRecord(model.state_dict()),
                "metrics": MetricRecord(
                    {
                        "train_loss": epoch_losses[-1],
                        "train_loss_mean": sum(epoch_losses) / len(epoch_losses),
                        "num-examples": len(trainloader.dataset),
                    }
                ),
            }
        ),
        reply_to=msg,
    )


@app.evaluate()
def evaluate(msg: Message, context: Context) -> Message:
    """Evaluate the global PaperCNN on one client's held-out 20% split."""
    model = PaperCNN()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    _, testloader = _client_data(context)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metrics = evaluate_model(model, testloader, device)

    return Message(
        content=RecordDict(
            {
                "metrics": MetricRecord(
                    {
                        "eval_loss": metrics["loss"],
                        "eval_acc": metrics["accuracy"],
                        "num-examples": len(testloader.dataset),
                    }
                )
            }
        ),
        reply_to=msg,
    )
