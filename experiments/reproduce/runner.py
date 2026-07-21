"""Run one end-to-end Alzheimer MRI Flower reproduction experiment.

This runner connects ``experiments.reproduce.server:app`` to
``experiments.reproduce.client:app`` through Flower's local Ray simulation.
It is independent of where it runs, so the same command works locally and on a
Lightning.ai pod.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from metricdp_pytorch.runtime import RUN_CONFIG_ENV
from metricdp_pytorch.strategy_factory import AGGREGATION_METHODS, PRIVACY_MODES

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        default="homogeneous",
        help="partition mode interpreted by the selected data module",
    )
    parser.add_argument("--privacy", choices=PRIVACY_MODES, default="vanilla")
    parser.add_argument(
        "--aggregation", choices=AGGREGATION_METHODS, default="fedavg"
    )
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--fraction-evaluate", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-multiplier", type=float, default=0.01)
    parser.add_argument("--clipping-norm", type=float, default=5.0)
    parser.add_argument(
        "--partition-profile",
        default="auto",
        help="optional profile interpreted by the selected data module",
    )
    parser.add_argument(
        "--client-weights",
        default="",
        help="comma-separated non-IID quantity weights, one per client",
    )
    parser.add_argument("--initialization-epochs", type=int, default=20)
    parser.add_argument("--initialization-batch-size", type=int, default=32)
    parser.add_argument("--initialization-learning-rate", type=float, default=0.001)
    parser.add_argument("--max-client-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument(
        "--data-module",
        default="experiments.reproduce.dataset.alzheimer:create_data_module",
        help="pluggable dataset factory in package.module:factory format",
    )
    parser.add_argument(
        "--data-cache-dir",
        default="",
        help="optional cache directory passed to the data-module factory",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/reproduce"))
    parser.add_argument("--run-name")
    parser.add_argument("--save-model", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="one round/epoch with capped data (32 client, 64 server samples)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--max-parallel-clients",
        type=int,
        default=2,
        help="cap simultaneous Ray actors to control memory use",
    )
    parser.add_argument("--client-cpus", type=float, default=1.0)
    parser.add_argument(
        "--client-gpus",
        type=float,
        default=0.0,
        help="Ray GPUs per client actor, e.g. 0.25; default clients run on CPU",
    )
    parser.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    return parser


def _project_defaults() -> dict[str, Any]:
    project = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    return dict(project["tool"]["flwr"]["app"]["config"])


def _validate(args: argparse.Namespace) -> None:
    if args.num_clients < 2:
        raise ValueError("num-clients must be at least two.")
    if not 0.0 < args.fraction_evaluate <= 1.0:
        raise ValueError("fraction-evaluate must be in (0, 1].")
    if args.rounds < 1 or args.local_epochs < 1:
        raise ValueError("rounds and local-epochs must be positive.")
    if args.batch_size < 1 or args.initialization_batch_size < 1:
        raise ValueError("batch sizes must be positive.")
    if args.initialization_epochs < 1:
        raise ValueError("initialization-epochs must be positive.")
    if args.max_parallel_clients < 1:
        raise ValueError("max-parallel-clients must be positive.")
    if args.client_cpus <= 0 or args.client_gpus < 0:
        raise ValueError("client-cpus must be positive and client-gpus non-negative.")
    if args.client_weights:
        weights = [part.strip() for part in args.client_weights.split(",") if part.strip()]
        if len(weights) != args.num_clients:
            raise ValueError("client-weights must contain one value per client.")
        if any(float(weight) < 0 for weight in weights):
            raise ValueError("client-weights must be non-negative.")


def build_run_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge CLI choices into the project's Flower run-config defaults."""
    _validate(args)
    rounds = 1 if args.smoke else args.rounds
    local_epochs = 1 if args.smoke else args.local_epochs
    initialization_epochs = 1 if args.smoke else args.initialization_epochs
    max_client_samples = (
        32 if args.smoke and args.max_client_samples == 0 else args.max_client_samples
    )
    max_test_samples = (
        64 if args.smoke and args.max_test_samples == 0 else args.max_test_samples
    )
    run_name = args.run_name or (
        f"paper__{args.partition}__{args.privacy}__{args.aggregation}"
        f"__clients-{args.num_clients}__seed-{args.seed}"
    )
    output_dir = args.output_dir.resolve()
    cache_dir = str(args.data_cache_dir).strip()
    if cache_dir:
        cache_dir = str(Path(cache_dir).expanduser().resolve())

    config = _project_defaults()
    config.update(
        {
            "partition-mode": args.partition,
            "partition-profile": args.partition_profile,
            "client-weights": args.client_weights,
            "privacy": args.privacy,
            "aggregation": args.aggregation,
            "seed": args.seed,
            "num-clients": args.num_clients,
            "fraction-evaluate": args.fraction_evaluate,
            "num-server-rounds": rounds,
            "local-epochs": local_epochs,
            "learning-rate": args.learning_rate,
            "batch-size": args.batch_size,
            "initialization-batch-size": args.initialization_batch_size,
            "initialization-epochs": initialization_epochs,
            "initialization-learning-rate": args.initialization_learning_rate,
            "max-client-samples": max_client_samples,
            "max-test-samples": max_test_samples,
            "noise-multiplier": args.noise_multiplier,
            "clipping-norm": args.clipping_norm,
            "data-module": args.data_module,
            "data-cache-dir": cache_dir,
            "output-dir": str(output_dir),
            "run-name": run_name,
            "save-model": args.save_model,
        }
    )
    return config


def _run_worker(
    config: dict[str, Any],
    *,
    num_supernodes: int,
    max_parallel_clients: int,
    client_cpus: float,
    client_gpus: float,
    verbose: bool,
) -> None:
    os.environ[RUN_CONFIG_ENV] = json.dumps(config)

    from flwr.simulation import run_simulation

    from experiments.reproduce.client import app as client_app
    from experiments.reproduce.server import app as server_app

    parallel_clients = min(num_supernodes, max_parallel_clients)
    total_cpus = max(1, math.ceil(parallel_clients * client_cpus))
    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=num_supernodes,
        backend_config={
            "init_args": {"num_cpus": total_cpus},
            "client_resources": {
                "num_cpus": client_cpus,
                "num_gpus": client_gpus,
            },
        },
        verbose_logging=verbose,
    )


def _launch_isolated(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Re-exec through a no-space venv path so Ray works from spaced paths."""
    with tempfile.TemporaryDirectory(prefix="paper-reproduction-") as temporary:
        temporary_dir = Path(temporary)
        config_path = temporary_dir / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        venv_link = temporary_dir / "venv"
        os.symlink(sys.prefix, venv_link, target_is_directory=True)
        command = [
            str(venv_link / "bin" / "python"),
            "-m",
            "experiments.reproduce.runner",
            "--worker-config",
            str(config_path),
            "--num-clients",
            str(args.num_clients),
            "--max-parallel-clients",
            str(args.max_parallel_clients),
            "--client-cpus",
            str(args.client_cpus),
            "--client-gpus",
            str(args.client_gpus),
        ]
        if args.verbose:
            command.append("--verbose")
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _print_result(config: dict[str, Any]) -> None:
    result_path = Path(config["output-dir"]) / f"{config['run-name']}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    history = result["server_evaluate_metrics"]
    completed = [int(round_number) for round_number in history if int(round_number) > 0]
    final_metrics = history[str(max(completed))] if completed else history.get("0", {})
    print(f"Result: {result_path}")
    if final_metrics:
        print(f"Final server metrics: {final_metrics}")


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.worker_config is not None:
        config = json.loads(args.worker_config.read_text(encoding="utf-8"))
        _run_worker(
            config,
            num_supernodes=args.num_clients,
            max_parallel_clients=args.max_parallel_clients,
            client_cpus=args.client_cpus,
            client_gpus=args.client_gpus,
            verbose=args.verbose,
        )
        return

    try:
        config = build_run_config(args)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(config, indent=2), flush=True)
    if args.dry_run:
        print("Dry run only; remove --dry-run to start the simulation.")
        return
    Path(config["output-dir"]).mkdir(parents=True, exist_ok=True)
    _launch_isolated(args, config)
    _print_result(config)


if __name__ == "__main__":
    main()
