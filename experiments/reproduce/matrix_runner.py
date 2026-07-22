"""Run a configurable partition × privacy × aggregation reproduction matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np

from metricdp_pytorch.strategy_factory import AGGREGATION_METHODS, PRIVACY_MODES


@dataclass(frozen=True)
class MatrixRun:
    """One independently executable reproduction configuration."""

    partition: str
    privacy: str
    aggregation: str
    seed: int

    @property
    def name(self) -> str:
        return (
            f"paper__{self.partition}__{self.privacy}__{self.aggregation}"
            f"__seed-{self.seed}"
        )


def build_matrix(
    *,
    partitions: list[str],
    privacy_modes: list[str],
    aggregations: list[str],
    seeds: list[int],
) -> list[MatrixRun]:
    """Build the deterministic Cartesian experiment matrix."""
    return [
        MatrixRun(partition, privacy, aggregation, seed)
        for partition, privacy, aggregation, seed in product(
            partitions,
            privacy_modes,
            aggregations,
            seeds,
        )
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--noise-multiplier", type=float, default=0.01)
    parser.add_argument("--clipping-norm", type=float, default=5.0)
    parser.add_argument("--initialization-epochs", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("results-reproduce-step1/matrix"))
    parser.add_argument("--parallel-experiments", type=int, default=1)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-parallel-clients", type=int, default=2)
    parser.add_argument("--client-cpus", type=float, default=1.0)
    parser.add_argument("--client-gpus", type=float, default=0.0)
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _artifact_paths(output_dir: Path, run: MatrixRun) -> dict[str, Path]:
    return {
        "run_json": output_dir / f"{run.name}.json",
        "evaluation_json": output_dir / f"{run.name}.evaluation.json",
        "predictions": output_dir / f"{run.name}.predictions.npz",
        "log": output_dir / "logs" / f"{run.name}.log",
    }


def validate_artifacts(output_dir: Path, run: MatrixRun) -> bool:
    """Validate the three final artifacts required for a completed run."""
    paths = _artifact_paths(output_dir, run)
    try:
        history = json.loads(paths["run_json"].read_text(encoding="utf-8"))
        evaluation = json.loads(
            paths["evaluation_json"].read_text(encoding="utf-8")
        )
        rounds = [
            int(round_number)
            for round_number in history["server_evaluate_metrics"]
            if int(round_number) > 0
        ]
        recorded_accuracy = float(
            history["server_evaluate_metrics"][str(max(rounds))]["accuracy"]
        )
        if evaluation["run_name"] != run.name:
            return False
        if abs(evaluation["server_final_test"]["accuracy"] - recorded_accuracy) >= 1e-12:
            return False
        with np.load(paths["predictions"]) as predictions:
            return (
                len(predictions["server_y_true"])
                == evaluation["server_final_test"]["num_examples"]
                and predictions["server_probabilities"].shape[1] == 4
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def build_command(args: argparse.Namespace, run: MatrixRun) -> list[str]:
    """Build the single-run subprocess command for one matrix entry."""
    return [
        sys.executable,
        "-m",
        "experiments.reproduce.runner",
        "--partition",
        run.partition,
        "--privacy",
        run.privacy,
        "--aggregation",
        run.aggregation,
        "--seed",
        str(run.seed),
        "--rounds",
        str(args.rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--noise-multiplier",
        str(args.noise_multiplier),
        "--clipping-norm",
        str(args.clipping_norm),
        "--initialization-epochs",
        str(args.initialization_epochs),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--run-name",
        run.name,
        "--max-parallel-clients",
        str(args.max_parallel_clients),
        "--client-cpus",
        str(args.client_cpus),
        "--client-gpus",
        str(args.client_gpus),
    ]


def _execute_one(args: argparse.Namespace, run: MatrixRun) -> dict[str, object]:
    paths = _artifact_paths(args.output_dir, run)
    if not args.rerun_completed and validate_artifacts(args.output_dir, run):
        return {**asdict(run), "run_name": run.name, "status": "skipped", "seconds": 0}

    command = build_command(args, run)
    started = time.monotonic()
    with paths["log"].open("w", encoding="utf-8") as log:
        log.write("COMMAND=" + subprocess.list2cmdline(command) + "\n")
        log.flush()
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    seconds = round(time.monotonic() - started, 3)
    valid = completed.returncode == 0 and validate_artifacts(args.output_dir, run)
    return {
        **asdict(run),
        "run_name": run.name,
        "status": "passed" if valid else "failed",
        "return_code": completed.returncode,
        "seconds": seconds,
        "log": str(paths["log"]),
    }


def _run_pass(
    args: argparse.Namespace,
    runs: list[MatrixRun],
) -> list[dict[str, object]]:
    with ThreadPoolExecutor(max_workers=args.parallel_experiments) as executor:
        futures = {executor.submit(_execute_one, args, run): run for run in runs}
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"{result['status'].upper():7} {result['run_name']} "
                f"({result['seconds']}s)",
                flush=True,
            )
        return results


def main() -> None:
    args = _parser().parse_args()
    if args.parallel_experiments < 1 or args.retries < 0:
        raise SystemExit("parallel-experiments must be positive and retries non-negative")
    args.output_dir = args.output_dir.resolve()
    (args.output_dir / "logs").mkdir(parents=True, exist_ok=True)
    matrix = build_matrix(
        partitions=args.partitions,
        privacy_modes=args.privacy_modes,
        aggregations=args.aggregations,
        seeds=args.seeds,
    )
    print(f"Matrix contains {len(matrix)} configurations.")
    if args.dry_run:
        for run in matrix:
            print(subprocess.list2cmdline(build_command(args, run)))
        return

    all_attempts: list[dict[str, object]] = []
    pending = matrix
    for attempt in range(args.retries + 1):
        if not pending:
            break
        print(f"Starting pass {attempt + 1} with {len(pending)} configurations.")
        all_attempts.extend(_run_pass(args, pending))
        pending = [run for run in matrix if not validate_artifacts(args.output_dir, run)]

    manifest = {
        "configuration_count": len(matrix),
        "complete_count": len(matrix) - len(pending),
        "failed_runs": [run.name for run in pending],
        "attempts": all_attempts,
    }
    (args.output_dir / "matrix-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Complete: {manifest['complete_count']}/{manifest['configuration_count']}; "
        f"failed: {len(pending)}"
    )
    raise SystemExit(1 if pending else 0)


if __name__ == "__main__":
    main()
