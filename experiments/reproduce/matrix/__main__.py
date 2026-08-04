"""Run a configurable partition × privacy × aggregation reproduction matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.reproduce.matrix import Hyperparams, Matrix, is_complete, run_combos
from metricdp_pytorch.strategy_factory import AGGREGATION_METHODS, PRIVACY_MODES


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--name-prefix", type=str)
    parser.add_argument(
        "--data-module",
        required=True,
        help="data-module factory in package.module:factory format",
    )

    parser.add_argument(
        "--model-module",
        required=True,
        help="model factory in package.module:factory format",
    )

    parser.add_argument(
        "--partitions",
        nargs="+",
        default=["homogeneous", "non-iid"],
    )
    parser.add_argument(
        "--privacy-modes",
        nargs="+",
        choices=PRIVACY_MODES,
        default=list(PRIVACY_MODES),
    )
    parser.add_argument(
        "--aggregations",
        nargs="+",
        choices=AGGREGATION_METHODS,
        default=list(AGGREGATION_METHODS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--num-clients", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--noise-multiplier", nargs="+", type=float, default=[0.01])
    parser.add_argument("--clipping-norm", type=float, default=5.0)
    parser.add_argument("--initialization-epochs", type=int, default=20)
    parser.add_argument("--parallel-experiments", type=int, default=1)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-parallel-clients", type=int, default=2)
    parser.add_argument("--client-cpus", type=float, default=1.0)
    parser.add_argument("--rerun-completed", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.parallel_experiments < 1 or args.retries < 0:
        raise SystemExit("parallel-experiments must be positive and retries non-negative")
    if args.num_clients < 1:
        raise SystemExit("num-clients must be positive")
    args.output_dir = args.output_dir.resolve()
    matrix = Matrix(
        partitions=tuple(args.partitions),
        privacy_modes=tuple(args.privacy_modes),
        aggregations=tuple(args.aggregations),
        seeds=tuple(args.seeds),
        noise_multipliers=tuple(args.noise_multiplier),
        hyperparams=Hyperparams(
            clipping_norm=args.clipping_norm,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            initialization_epochs=args.initialization_epochs,
        ),
        data_module=args.data_module,
        model_module=args.model_module,
    ).list_combos(name_prefix=args.name_prefix, num_clients=args.num_clients)
    print(f"Matrix contains {len(matrix)} configurations.")
    pending = matrix
    for attempt in range(args.retries + 1):
        if not pending:
            break
        print(f"Starting pass {attempt + 1} with {len(pending)} configurations.")
        run_combos(
            pending,
            output_dir=args.output_dir,
            parallel_experiments=args.parallel_experiments,
            max_parallel_clients=args.max_parallel_clients,
            force=args.rerun_completed,
            client_cpus=args.client_cpus,
        )
        pending = [
            run
            for run in matrix
            if not is_complete(
                run.result_path(args.output_dir),
                expected_rounds=run.hyperparams.rounds,
            )
        ]

    print(f"Complete: {len(matrix) - len(pending)}/{len(matrix)}; failed: {len(pending)}")
    raise SystemExit(1 if pending else 0)


if __name__ == "__main__":
    main()
