"""Flower ServerApp for the paper reproduction's utility experiments.

This module deliberately has no client-inference-attack roles, shadow data, or
IN/OUT runs. It selects one paper strategy, starts federated training, and can
optionally persist the Flower metric histories.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import Result

from experiments.reproduce.detailed_evaluation import evaluate_state_dict
from experiments.reproduce.paper_loss import make_evaluate_fn
from experiments.reproduce.paper_strategies import create_paper_strategy
from experiments.reproduce.paper_training import create_initial_model
from metricdp_pytorch.data_module import load_data_module
from metricdp_pytorch.model_module import load_model
from metricdp_pytorch.utils.device import resolve_device
from metricdp_pytorch.utils.runtime import runtime_config

app = ServerApp()



def _plain_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Convert Flower metric values into JSON-compatible Python scalars."""
    return {
        str(key): value.item() if hasattr(value, "item") else value
        for key, value in metrics.items()
    }


def _runtime_metadata() -> dict[str, Any]:
    """Return reproducibility metadata captured by the process doing the run."""
    packages = {}
    for package in ("flwr", "numpy", "scikit-learn", "torch", "torchvision"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "unavailable"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        commit, dirty = "unavailable", None

    device = resolve_device()
    device_name = str(device)
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": packages,
        "device": str(device),
        "device_name": device_name,
        "git_commit": commit,
        "git_dirty": dirty,
        "python_executable": sys.executable,
    }


def checkpoint_model_path(output_dir: Path, run_name: str, round_number: int) -> Path:
    """Return the model-checkpoint path for one aggregated server round."""
    return output_dir / f"{run_name}.round-{round_number}.pt"


def _checkpointing_evaluate_fn(
    evaluate_fn: Callable[[int, ArrayRecord], MetricRecord],
    *,
    output_dir: Path,
    run_name: str,
    checkpoint_rounds: set[int],
) -> Callable[[int, ArrayRecord], MetricRecord]:
    """Decorate server evaluation to persist selected round models."""

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        if server_round in checkpoint_rounds:
            path = checkpoint_model_path(output_dir, run_name, server_round)
            torch.save(arrays.to_torch_state_dict(), path)
            print(f"Round {server_round} checkpoint written to {path}")
        return evaluate_fn(server_round, arrays)

    return evaluate


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
    this function falls back to a randomly initialized configured model.
    """
    if initial_arrays is None:
        initial_arrays = ArrayRecord(
            load_model(str(config["model-module"])).state_dict()
        )

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
    model_factory = lambda: load_model(str(config["model-module"]))
    initial_model, initialization_losses = create_initial_model(
        str(config["aggregation"]),
        validation_loader,
        seed=seed,
        epochs=int(config.get("initialization-epochs", 20)),
        learning_rate=float(config.get("initialization-learning-rate", 1e-3)),
        model_factory=model_factory,
    )

    output_dir = config.get("output-dir")
    run_name = config.get("run-name")
    evaluate_fn = make_evaluate_fn(
        final_testloader, model_factory=model_factory
    )
    checkpoint_rounds = {
        int(round_number) for round_number in config.get("checkpoint-rounds", [])
    }
    if checkpoint_rounds:
        if not output_dir or not run_name:
            raise ValueError("Checkpoint rounds require output-dir and run-name.")
        destination = Path(str(output_dir))
        destination.mkdir(parents=True, exist_ok=True)
        evaluate_fn = _checkpointing_evaluate_fn(
            evaluate_fn,
            output_dir=destination,
            run_name=str(run_name),
            checkpoint_rounds=checkpoint_rounds,
        )

    result = run(
        grid,
        config,
        evaluate_fn=evaluate_fn,
        initial_arrays=ArrayRecord(initial_model.state_dict()),
    )

    if output_dir and run_name:
        destination = Path(str(output_dir))
        destination.mkdir(parents=True, exist_ok=True)
        metadata = {
            "run_name": str(run_name),
            "data_module": str(config["data-module"]),
            "model_module": str(config["model-module"]),
            "partition_mode": str(config.get("partition-mode", "unknown")),
            "partition_profile": str(config.get("partition-profile", "auto")),
            "privacy": str(config["privacy"]),
            "aggregation": str(config["aggregation"]),
            "seed": int(config.get("seed", 0)),
            "num_clients": int(config["num-clients"]),
            "rounds": int(config["num-server-rounds"]),
            "local_epochs": int(config.get("local-epochs", 5)),
            "batch_size": int(config.get("batch-size", 32)),
            "initialization_batch_size": int(
                config.get("initialization-batch-size", 32)
            ),
            "max_client_samples": int(config.get("max-client-samples", 0)),
            "max_test_samples": int(config.get("max-test-samples", 0)),
            "client_weights": str(config.get("client-weights", "")),
            "noise_multiplier": float(config.get("noise-multiplier", 0.01)),
            "clipping_norm": float(config.get("clipping-norm", 5.0)),
            "initialization_pretrained": bool(initialization_losses),
            "initialization_epochs": len(initialization_losses),
            "initialization_final_loss": (
                initialization_losses[-1] if initialization_losses else None
            ),
            **_runtime_metadata(),
        }
        path = destination / f"{run_name}.json"
        serialized_result = result_to_dict(result, metadata)
        path.write_text(
            json.dumps(serialized_result, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Experiment result written to {path}")

        evaluation_path = destination / f"{run_name}.evaluation.json"
        predictions_path = destination / f"{run_name}.predictions.npz"
        evaluate_state_dict(
            state_dict=result.arrays.to_torch_state_dict(),
            run=serialized_result,
            run_json_path=path,
            evaluation_json_path=evaluation_path,
            predictions_path=predictions_path,
            data_module=data_module,
            model=model_factory(),
            class_names=getattr(data_module, "class_names", ()),
        )
        print(
            "Detailed evaluation written to "
            f"{evaluation_path} and {predictions_path}"
        )
