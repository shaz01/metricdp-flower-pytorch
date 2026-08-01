"""Orchestrate the 48-client Client Inference Attack experiment.

For each of 12 ``(partition_mode, privacy, aggregation)`` combinations, run
two timing variants:

- ``first-round``: mirrors ``experiments/cia/runner.py``'s paper-exact
  methodology (1 round, local-epochs=20), scaled from 3 to 48 clients.
- ``post-convergence``: mirrors ``experiments/client_scaling/
  sweep_48_clients.py``'s actual training regime (20 rounds,
  local-epochs=5, noise-multiplier=0.05), retaining its final-round checkpoint.

Both shell out to the existing, unmodified ``experiments.reproduce.runner``
CLI with the default Alzheimer data module, then evaluate the resulting
saved model's loss on the global test set and on a fixed target client's
(``partition_id=0``) shadow split, reporting the relative-difference attack
score for each combination.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from experiments.cia_client_scaling.result import CiaScalingResult, make_cia_scaling_result
from experiments.cia_client_scaling.shadow import target_shadow_loader
from experiments.reproduce.dataset.alzheimer import AlzheimerDataModule
from experiments.reproduce.matrix import Combo, Hyperparams
from experiments.reproduce.matrix.run_combo import run_one_combo
from experiments.reproduce.paper_cnn import PaperCNN
from experiments.reproduce.paper_loss import evaluate_model
from metricdp_pytorch.strategy_factory import PRIVACY_MODES

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NUM_CLIENTS = 48
TARGET_PARTITION_ID = 0
SEED = 42
BATCH_SIZE = 32
PARTITION_MODES = ("homogeneous", "non-iid")
AGGREGATIONS = ("fedavg", "fedyogi")

TIMING_CONFIGS: dict[str, dict[str, int | float]] = {
    "first-round": {
        "rounds": 1,
        "local_epochs": 20,
        "noise_multiplier": 0.01,
        "clipping_norm": 5.0,
    },
    "post-convergence": {
        "rounds": 20,
        "local_epochs": 5,
        "noise_multiplier": 0.05,
        "clipping_norm": 5.0,
    },
}
TIMINGS = tuple(TIMING_CONFIGS)


def resolve_noise_multiplier(timing: str, noise_multiplier: float | None) -> float:
    """Return the noise multiplier to train with for ``timing``.

    ``None`` keeps the timing's published default, so the paper-faithful
    hyperparameters stay the default behaviour.
    """
    if noise_multiplier is not None:
        return noise_multiplier
    return float(TIMING_CONFIGS[timing]["noise_multiplier"])


def build_combo(
        *,
        partition_mode: str,
        timing: str,
        privacy: str,
        aggregation: str,
        noise_multiplier: float | None = None,
) -> Combo:
    """Represent one CIA training run using the shared reproduction matrix."""
    timing_config = TIMING_CONFIGS[timing]
    return Combo(
        name_prefix=f"cia_scaling__{timing}",
        num_clients=NUM_CLIENTS,
        partition=partition_mode,
        privacy=privacy,
        aggregation=aggregation,
        seed=SEED,
        noise_multiplier=resolve_noise_multiplier(timing, noise_multiplier),
        hyperparams=Hyperparams(
            clipping_norm=float(timing_config["clipping_norm"]),
            rounds=int(timing_config["rounds"]),
            local_epochs=int(timing_config["local_epochs"]),
            batch_size=BATCH_SIZE,
            learning_rate=0.001,
            initialization_epochs=20,
        ),
        data_module="experiments.reproduce.dataset.alzheimer:create_data_module",
    )


def _log(log_path: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def evaluate_combo(
        model_path: Path,
        *,
        partition_mode: str,
        device: torch.device,
) -> tuple[float, float, int]:
    """Return ``(aggregated_test_loss, target_shadow_loss, shadow_size)`` for
    one saved model."""
    model = PaperCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    data_module = AlzheimerDataModule()
    _validation_loader, test_loader = data_module.server_loaders(
        batch_size=BATCH_SIZE, seed=SEED
    )
    shadow_loader = target_shadow_loader(
        target_partition_id=TARGET_PARTITION_ID,
        num_partitions=NUM_CLIENTS,
        partition_mode=partition_mode,
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    aggregated_metrics = evaluate_model(model, test_loader, device)
    target_metrics = evaluate_model(model, shadow_loader, device)
    shadow_size = len(shadow_loader.dataset)
    return aggregated_metrics["loss"], target_metrics["loss"], shadow_size


def _load_existing_report(
        report_path: Path,
) -> tuple[dict[tuple[str, str, str, str], dict], list[str]]:
    """Load a prior run's report (if any) so a new invocation merges into it
    instead of clobbering it. Returns (results_by_key, failed)."""
    if not report_path.exists():
        return {}, []
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}, []
    results_by_key = {
        (
            row["timing"],
            row["partition_mode"],
            row["privacy"],
            row["aggregation"],
            row.get("noise_multiplier"),
        ): row
        for row in data.get("results", [])
    }
    return results_by_key, list(data.get("failed", []))


def run_cia_client_scaling(
        *,
        output_dir: Path,
        partition_modes: tuple[str, ...] = PARTITION_MODES,
        privacy_modes: tuple[str, ...] = PRIVACY_MODES,
        aggregations: tuple[str, ...] = AGGREGATIONS,
        timings: tuple[str, ...] = TIMINGS,
        max_parallel_clients: int = 4,
        force: bool = False,
        noise_multiplier: float | None = None,
) -> list[CiaScalingResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "sweep_progress.log"
    report_path = output_dir / "cia_client_scaling.json"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results_by_key, failed = _load_existing_report(report_path)

    def _write_report() -> None:
        report_path.write_text(
            json.dumps(
                {"results": list(results_by_key.values()), "failed": failed}, indent=2
            )
            + "\n",
            encoding="utf-8",
        )

    total = len(partition_modes) * len(privacy_modes) * len(aggregations) * len(timings)
    completed = 0

    for timing in timings:
        for partition_mode in partition_modes:
            for privacy in privacy_modes:
                for aggregation in aggregations:
                    resolved_noise = resolve_noise_multiplier(timing, noise_multiplier)
                    combo = build_combo(
                        partition_mode=partition_mode,
                        timing=timing,
                        privacy=privacy,
                        aggregation=aggregation,
                        noise_multiplier=noise_multiplier,
                    )
                    name = combo.run_name()
                    checkpoint_round = combo.hyperparams.rounds
                    model_path = output_dir / f"{name}.round-{checkpoint_round}.pt"
                    key = (
                        timing,
                        partition_mode,
                        privacy,
                        aggregation,
                        resolved_noise,
                    )
                    success = run_one_combo(
                        combo,
                        output_dir=output_dir,
                        max_parallel_clients=max_parallel_clients,
                        force=force,
                        log=lambda message: _log(log_path, message),
                        checkpoint_rounds=(checkpoint_round,),
                    )
                    completed += 1
                    if not success:
                        if name not in failed:
                            failed.append(name)
                        _log(log_path, f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")
                        _write_report()
                        continue

                    try:
                        aggregated_loss, target_loss, shadow_size = evaluate_combo(
                            model_path, partition_mode=partition_mode, device=device
                        )
                        result = make_cia_scaling_result(
                            partition_mode=partition_mode,
                            timing=timing,
                            privacy=privacy,
                            aggregation=aggregation,
                            noise_multiplier=resolved_noise,
                            aggregated_test_loss=aggregated_loss,
                            target_shadow_loss=target_loss,
                            shadow_size=shadow_size,
                        )
                    except Exception as error:  # noqa: BLE001 - log and continue the sweep
                        _log(log_path, f"FAILED {name} (evaluation error: {error})")
                        if name not in failed:
                            failed.append(name)
                        _log(log_path, f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")
                        _write_report()
                        continue

                    results_by_key[key] = result.__dict__
                    if name in failed:
                        failed.remove(name)
                    _log(log_path, f"EVALUATED {name}")
                    _log(log_path, f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")
                    _write_report()

    if failed:
        _log(log_path, "Failed combinations: " + ", ".join(failed))
    return [CiaScalingResult(**row) for row in results_by_key.values()]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "cia_client_scaling",
    )
    parser.add_argument(
        "--max-parallel-clients",
        type=int,
        default=4,
        help=(
            "cap simultaneous Ray actors. Passed straight to the reproduce "
            "runner, where it also sets each actor's auto-detected GPU share; "
            "Ray's num_gpus never reserves VRAM, so setting this near "
            f"--num-clients ({NUM_CLIENTS}) can exhaust the device and OOM"
        ),
    )
    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=None,
        help=(
            "override the timing's noise multiplier; default keeps the published "
            "per-timing value (first-round 0.01, post-convergence 0.05). Averaging "
            "more clients lowers each client's sensitivity, so DP injects less "
            "noise per parameter as the cohort grows: at 48 clients the published "
            "3-client defaults leave global-dp and metric-privacy converging to "
            "essentially vanilla. Raise this to push the arms apart far enough to "
            "measure a difference"
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun every combination even if a complete result already exists",
    )
    parser.add_argument(
        "--timings",
        default=",".join(TIMINGS),
        help="comma-separated subset of timings to run",
    )
    parser.add_argument(
        "--partitions",
        default=",".join(PARTITION_MODES),
        help="comma-separated subset of partition modes to run",
    )
    parser.add_argument(
        "--aggregations",
        default=",".join(AGGREGATIONS),
        help="comma-separated subset of aggregation methods to run",
    )
    parser.add_argument(
        "--privacy",
        default=",".join(PRIVACY_MODES),
        help="comma-separated subset of privacy modes to run",
    )
    return parser


def parse_subset(
        parser: argparse.ArgumentParser,
        raw: str,
        valid: tuple[str, ...],
        flag: str,
) -> tuple[str, ...]:
    """Parse one comma-separated CLI subset, erroring on empty or unknown values.

    Order follows the caller's list, so the sweep runs in the order requested
    rather than in declaration order.
    """
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        parser.error(f"{flag} must name at least one value; choose from {list(valid)}")
    invalid = [value for value in values if value not in valid]
    if invalid:
        parser.error(
            f"invalid {flag} value(s) {invalid!r}; must be a subset of {list(valid)}"
        )
    return values


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    timings = parse_subset(parser, args.timings, TIMINGS, "--timings")
    partition_modes = parse_subset(parser, args.partitions, PARTITION_MODES, "--partitions")
    aggregations = parse_subset(parser, args.aggregations, AGGREGATIONS, "--aggregations")
    privacy_modes = parse_subset(parser, args.privacy, PRIVACY_MODES, "--privacy")
    results = run_cia_client_scaling(
        output_dir=args.output_dir,
        timings=timings,
        partition_modes=partition_modes,
        aggregations=aggregations,
        privacy_modes=privacy_modes,
        max_parallel_clients=args.max_parallel_clients,
        force=args.force,
        noise_multiplier=args.noise_multiplier,
    )
    for result in results:
        print(
            f"{result.timing:16s} {result.partition_mode:12s} {result.privacy:15s} "
            f"{result.aggregation:8s} agg={result.aggregated_test_loss:.3f} "
            f"target={result.target_shadow_loss:.3f} shadow_n={result.shadow_size} "
            f"diff={result.difference_pct:.3f}%"
        )


if __name__ == "__main__":
    main()
