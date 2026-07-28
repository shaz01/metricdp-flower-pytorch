"""Orchestrate the 48-client Client Inference Attack experiment.

For each of 12 ``(partition_mode, privacy, aggregation)`` combinations, run
two timing variants:

- ``first-round``: mirrors ``experiments/cia/runner.py``'s paper-exact
  methodology (1 round, local-epochs=20), scaled from 3 to 48 clients.
- ``post-convergence``: mirrors ``experiments/client_scaling/
  sweep_48_clients.py``'s actual training regime (20 rounds,
  local-epochs=5, noise-multiplier=0.05), with ``--save-model`` added.

Both shell out to the existing, unmodified ``experiments.reproduce.runner``
CLI with the default Alzheimer data module, then evaluate the resulting
saved model's loss on the global test set and on a fixed target client's
(``partition_id=0``) shadow split, reporting the relative-difference attack
score for each combination.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

from experiments.cia_client_scaling.result import CiaScalingResult, make_cia_scaling_result
from experiments.cia_client_scaling.shadow import target_shadow_loader
from experiments.reproduce.dataset.alzheimer import AlzheimerDataModule
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


def format_noise(noise_multiplier: float) -> str:
    """Render a noise multiplier as a filename-safe token, e.g. 0.12 -> '0p12'."""
    return f"{noise_multiplier:g}".replace(".", "p")


def run_name(
    partition_mode: str,
    timing: str,
    privacy: str,
    aggregation: str,
    *,
    noise_multiplier: float | None = None,
) -> str:
    """Build a deterministic run name.

    A non-default noise multiplier is encoded in the name so sweeps at several
    noise levels neither collide on disk nor skip each other via resumability.
    The timing's default value keeps the original unsuffixed name.
    """
    base = f"cia_scaling__{timing}__{partition_mode}__{privacy}__{aggregation}"
    resolved = resolve_noise_multiplier(timing, noise_multiplier)
    if resolved == float(TIMING_CONFIGS[timing]["noise_multiplier"]):
        return base
    return f"{base}__nm{format_noise(resolved)}"


def build_reproduce_command(
    *,
    partition_mode: str,
    timing: str,
    privacy: str,
    aggregation: str,
    output_dir: Path,
    max_parallel_clients: int,
    noise_multiplier: float | None = None,
) -> list[str]:
    """Build the argv for one real 48-client CIA training run."""
    timing_config = TIMING_CONFIGS[timing]
    resolved_noise = resolve_noise_multiplier(timing, noise_multiplier)
    name = run_name(
        partition_mode, timing, privacy, aggregation, noise_multiplier=noise_multiplier
    )
    return [
        sys.executable,
        "-m",
        "experiments.reproduce.runner",
        "--num-clients",
        str(NUM_CLIENTS),
        "--partition",
        partition_mode,
        "--privacy",
        privacy,
        "--aggregation",
        aggregation,
        "--rounds",
        str(timing_config["rounds"]),
        "--local-epochs",
        str(timing_config["local_epochs"]),
        "--noise-multiplier",
        str(resolved_noise),
        "--clipping-norm",
        str(timing_config["clipping_norm"]),
        "--seed",
        str(SEED),
        "--output-dir",
        str(output_dir),
        "--run-name",
        name,
        "--save-model",
        "--max-parallel-clients",
        str(max_parallel_clients),
    ]


def is_training_complete(path: Path, *, expected_rounds: int) -> bool:
    """Return whether ``path`` holds a valid, fully-completed training result.

    Treats a missing, unparseable, or short-of-rounds file as incomplete, so
    a prior run that was killed mid-write (or mid-sweep) is rerun rather than
    silently accepted. Mirrors
    ``experiments/client_scaling/sweep_48_clients.py``'s ``is_complete``.
    """
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    history = data.get("server_evaluate_metrics", {})
    completed_rounds = [
        int(round_number) for round_number in history if int(round_number) > 0
    ]
    return len(completed_rounds) >= expected_rounds


def _log(log_path: Path, message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_one_combo(
    *,
    partition_mode: str,
    timing: str,
    privacy: str,
    aggregation: str,
    output_dir: Path,
    max_parallel_clients: int,
    force: bool,
    noise_multiplier: float | None = None,
) -> tuple[Path, bool]:
    """Run one training combo unless already complete; return (model_path, success)."""
    name = run_name(
        partition_mode, timing, privacy, aggregation, noise_multiplier=noise_multiplier
    )
    result_path = output_dir / f"{name}.json"
    model_path = output_dir / f"{name}.pt"
    expected_rounds = int(TIMING_CONFIGS[timing]["rounds"])

    if (
        not force
        and is_training_complete(result_path, expected_rounds=expected_rounds)
        and model_path.exists()
    ):
        return model_path, True

    command = build_reproduce_command(
        partition_mode=partition_mode,
        timing=timing,
        privacy=privacy,
        aggregation=aggregation,
        output_dir=output_dir,
        max_parallel_clients=max_parallel_clients,
        noise_multiplier=noise_multiplier,
    )
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    return model_path, result.returncode == 0


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
                    name = run_name(
                        partition_mode,
                        timing,
                        privacy,
                        aggregation,
                        noise_multiplier=noise_multiplier,
                    )
                    key = (
                        timing,
                        partition_mode,
                        privacy,
                        aggregation,
                        resolved_noise,
                    )
                    _log(log_path, f"START {name}")
                    model_path, success = run_one_combo(
                        partition_mode=partition_mode,
                        timing=timing,
                        privacy=privacy,
                        aggregation=aggregation,
                        output_dir=output_dir,
                        max_parallel_clients=max_parallel_clients,
                        force=force,
                        noise_multiplier=noise_multiplier,
                    )
                    completed += 1
                    if not success:
                        _log(log_path, f"FAILED {name}")
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
                    _log(log_path, f"DONE {name}")
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
    parser.add_argument("--max-parallel-clients", type=int, default=4)
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
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    timings = tuple(part.strip() for part in args.timings.split(",") if part.strip())
    invalid = [timing for timing in timings if timing not in TIMINGS]
    if invalid:
        parser.error(
            f"invalid --timings value(s) {invalid!r}; must be a subset of {list(TIMINGS)}"
        )
    results = run_cia_client_scaling(
        output_dir=args.output_dir,
        timings=timings,
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
