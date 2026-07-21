"""Flower ServerApp for the paper reproduction's utility experiments.

This module deliberately has no client-inference-attack roles, shadow data, or
IN/OUT runs. It selects one paper strategy, starts federated training, and can
optionally persist the Flower metric histories.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import Result

from experiments.reproduce.paper_cnn import PaperCNN
from experiments.reproduce.paper_loss import make_evaluate_fn
from experiments.reproduce.paper_strategies import create_paper_strategy
from experiments.reproduce.paper_training import create_initial_model
from metricdp_pytorch.data_module import load_data_module
from metricdp_pytorch.utils.runtime import runtime_config

app = ServerApp()



def _plain_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Convert Flower metric values into JSON-compatible Python scalars."""
    return {
        str(key): value.item() if hasattr(value, "item") else value
        for key, value in metrics.items()
    }


def result_to_dict(result: Result, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize ordinary FL train/evaluation histories; no CIA fields."""

    def history(values: Mapping[int, MetricRecord]) -> dict[str, dict[str, Any]]:
        return {
            str(round_number): _plain_metrics(metrics)
            for round_number, metrics in sorted(values.items())
        }

    return {
        "metadata": dict(metadata),
        "train_metrics": history(result.train_metrics_clientapp),
        "client_evaluate_metrics": history(result.evaluate_metrics_clientapp),
        "server_evaluate_metrics": history(result.evaluate_metrics_serverapp),
    }


def run(
        grid: Grid,
        config: Mapping[str, Any],
        *,
        evaluate_fn: Callable[[int, ArrayRecord], MetricRecord] | None = None,
        initial_arrays: ArrayRecord | None = None,
) -> Result:
    """Run one ordinary paper utility/convergence experiment.

    ``main`` supplies the paper's validation-pretrained arrays for FedAvgM,
    FedOpt, and FedYogi. Direct callers may provide their own arrays; otherwise
    this function falls back to a randomly initialized ``PaperCNN``.
    """
    if initial_arrays is None:
        initial_arrays = ArrayRecord(PaperCNN().state_dict())

    strategy = create_paper_strategy(
        aggregation=str(config["aggregation"]),
        privacy=str(config["privacy"]),
        num_clients=int(config["num-clients"]),
        fraction_evaluate=float(config.get("fraction-evaluate", 1.0)),
        noise_multiplier=float(config.get("noise-multiplier", 0.01)),
        clipping_norm=float(config.get("clipping-norm", 5.0)),
    )
    return strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=ConfigRecord(
            {
                "lr": float(config["learning-rate"])
            }
        ),
        num_rounds=int(config["num-server-rounds"]),
        evaluate_fn=evaluate_fn,
    )


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run one reproduction server experiment without CIA evaluation."""
    config = runtime_config(context)
    seed = int(config.get("seed", 42))
    data_module = load_data_module(str(config["data-module"]), config)
    validation_loader, final_testloader = data_module.server_loaders(
        batch_size=int(config.get("initialization-batch-size", 32)),
        seed=seed,
        max_samples=int(config.get("max-test-samples", 0)),
    )
    initial_model, initialization_losses = create_initial_model(
        str(config["aggregation"]),
        validation_loader,
        seed=seed,
        epochs=int(config.get("initialization-epochs", 20)),
        learning_rate=float(config.get("initialization-learning-rate", 1e-3)),
    )
    result = run(
        grid,
        config,
        evaluate_fn=make_evaluate_fn(final_testloader),
        initial_arrays=ArrayRecord(initial_model.state_dict()),
    )

    output_dir = config.get("output-dir")
    run_name = config.get("run-name")
    save_model = bool(config.get("save-model", False))

    if output_dir and run_name:
        destination = Path(str(output_dir))
        destination.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_name": str(run_name),
            "data_module": str(config["data-module"]),
            "partition_mode": str(config.get("partition-mode", "unknown")),
            "partition_profile": str(config.get("partition-profile", "auto")),
            "privacy": str(config["privacy"]),
            "aggregation": str(config["aggregation"]),
            "seed": int(config.get("seed", 0)),
            "num_clients": int(config["num-clients"]),
            "rounds": int(config["num-server-rounds"]),
            "local_epochs": int(config.get("local-epochs", 5)),
            "noise_multiplier": float(config.get("noise-multiplier", 0.01)),
            "clipping_norm": float(config.get("clipping-norm", 5.0)),
            "initialization_pretrained": bool(initialization_losses),
            "initialization_epochs": len(initialization_losses),
            "initialization_final_loss": (
                initialization_losses[-1] if initialization_losses else None
            ),
        }
        path = destination / f"{run_name}.json"
        path.write_text(
            json.dumps(result_to_dict(result, metadata), indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        print(f"Experiment result written to {path}")

        if save_model:
            torch.save(
                result.arrays.to_torch_state_dict(),
                Path(str(output_dir)) / f"{run_name}.pt",
            )
