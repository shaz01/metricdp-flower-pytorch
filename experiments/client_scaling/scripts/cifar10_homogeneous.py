"""Accuracy-only homogeneous CIFAR-10 client-scaling sweep.

No CIA here: this sweep only measures accuracy/loss under homogeneous
partitioning for all three privacy modes, at whatever client count is passed on
the command line. The non-IID removal-adjacency CIA counterpart lives in
``experiments/cia/scripts/cifar10_remove.py``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.client_scaling.sweep_runner import run_sweep
from experiments.reproduce.matrix import Hyperparams, Matrix

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "client_scaling" / "cifar10_homogeneous"
)

DEFAULT_NUM_CLIENTS = 4

# TODO: try lowered local epoch and increased rounds
MATRIX = Matrix(
    partitions=("homogeneous",),
    privacy_modes=("vanilla", "global-dp", "metric-privacy"),
    aggregations=("fedavg",),
    seeds=(42,),
    noise_multipliers=(0.01,),  # TODO: calibrate per client count for CIFAR-10.
    hyperparams=Hyperparams(
        clipping_norm=5.0,
        rounds=20,
        local_epochs=5,
        batch_size=32,
        learning_rate=0.001,
        initialization_epochs=20,
    ),
    data_module="experiments.reproduce.dataset.cifar10:create_data_module",
    model_module="experiments.reproduce.cifar10_cnn:create_model",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clients",
        type=int,
        nargs="+",
        default=[DEFAULT_NUM_CLIENTS],
        help="One or more client counts to sweep.",
    )
    parser.add_argument(
        "--max-parallel-clients",
        type=int,
        help="Defaults to min(num_clients, 8).",  # TODO: tune per machine.
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    for num_clients in args.clients:
        if num_clients < 2:
            raise ValueError("num_clients must be at least 2.")
        output_dir = (args.output_dir / f"clients-{num_clients}").resolve()
        combos = MATRIX.list_combos(
            name_prefix="cifar10-homogeneous", num_clients=num_clients
        )
        run_sweep(
            combos,
            output_dir=output_dir,
            log_path=output_dir / "progress.log",
            max_parallel_clients=args.max_parallel_clients or min(num_clients, 8),
            force=args.force,
            start_message=(
                f"Sweep starting: {len(combos)} combinations, "
                f"num_clients={num_clients}"
            ),
        )


if __name__ == "__main__":
    main()
