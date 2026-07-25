"""Orchestrate the paper's first-round single-shot CIA experiment (Section 7.4.1).

For each of the 18 (privacy, aggregation) combinations, this launches one
real 1-round, 3-client Flower simulation by shelling out to the existing,
unmodified ``experiments.reproduce.runner`` CLI (pointed at this package's
``create_cia_data_module``), then evaluates the resulting saved model's loss
on the global test set and on the target client's shadow split, reporting
the relative-difference attack score for each combination.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from experiments.cia.attack import CiaResult, make_cia_result
from experiments.cia.dataset import CIA_NUM_CLIENTS, CiaDataModule
from experiments.reproduce.paper_cnn import PaperCNN
from experiments.reproduce.paper_loss import evaluate_model
from metricdp_pytorch.strategy_factory import AGGREGATION_METHODS, PRIVACY_MODES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CIA_LOCAL_EPOCHS = 20
CIA_SEED = 42
CIA_NOISE_MULTIPLIER = 0.01
CIA_CLIPPING_NORM = 5.0
CIA_BATCH_SIZE = 32


def run_name(privacy: str, aggregation: str) -> str:
    return f"cia__{privacy}__{aggregation}"


def build_reproduce_command(
    *,
    privacy: str,
    aggregation: str,
    output_dir: Path,
    max_parallel_clients: int,
) -> list[str]:
    """Build the argv for one real 1-round CIA training run.

    Reuses ``experiments.reproduce.runner`` unmodified, pointed at this
    package's ``create_cia_data_module`` factory instead of the paper's
    4-client module.
    """
    name = run_name(privacy, aggregation)
    return [
        sys.executable,
        "-m",
        "experiments.reproduce.runner",
        "--data-module",
        "experiments.cia.dataset:create_cia_data_module",
        "--num-clients",
        str(CIA_NUM_CLIENTS),
        "--rounds",
        "1",
        "--local-epochs",
        str(CIA_LOCAL_EPOCHS),
        "--privacy",
        privacy,
        "--aggregation",
        aggregation,
        "--seed",
        str(CIA_SEED),
        "--noise-multiplier",
        str(CIA_NOISE_MULTIPLIER),
        "--clipping-norm",
        str(CIA_CLIPPING_NORM),
        "--output-dir",
        str(output_dir),
        "--run-name",
        name,
        "--save-model",
        "--max-parallel-clients",
        str(max_parallel_clients),
    ]


def run_one_combo(
    *,
    privacy: str,
    aggregation: str,
    output_dir: Path,
    max_parallel_clients: int,
) -> Path:
    """Launch one real 1-round CIA training run; return the saved model path."""
    command = build_reproduce_command(
        privacy=privacy,
        aggregation=aggregation,
        output_dir=output_dir,
        max_parallel_clients=max_parallel_clients,
    )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return output_dir / f"{run_name(privacy, aggregation)}.pt"


def evaluate_combo(
    model_path: Path,
    *,
    data_module: CiaDataModule,
    device: torch.device,
) -> tuple[float, float]:
    """Return ``(aggregated_test_loss, target_shadow_loss)`` for one saved model."""
    model = PaperCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    _validation_loader, test_loader = data_module.server_loaders(
        batch_size=CIA_BATCH_SIZE, seed=CIA_SEED
    )
    shadow_loader = data_module.target_shadow_loader(
        batch_size=CIA_BATCH_SIZE, seed=CIA_SEED
    )

    aggregated_metrics = evaluate_model(model, test_loader, device)
    target_metrics = evaluate_model(model, shadow_loader, device)
    return aggregated_metrics["loss"], target_metrics["loss"]


def run_first_round_cia(
    *,
    output_dir: Path,
    aggregations: tuple[str, ...] = AGGREGATION_METHODS,
    privacy_modes: tuple[str, ...] = PRIVACY_MODES,
    max_parallel_clients: int = 2,
) -> list[CiaResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_module = CiaDataModule()

    results: list[CiaResult] = []
    for privacy in privacy_modes:
        for aggregation in aggregations:
            model_path = run_one_combo(
                privacy=privacy,
                aggregation=aggregation,
                output_dir=output_dir,
                max_parallel_clients=max_parallel_clients,
            )
            aggregated_loss, target_loss = evaluate_combo(
                model_path, data_module=data_module, device=device
            )
            results.append(
                make_cia_result(
                    privacy=privacy,
                    aggregation=aggregation,
                    aggregated_test_loss=aggregated_loss,
                    target_shadow_loss=target_loss,
                )
            )

    report_path = output_dir / "first_round_cia.json"
    report_path.write_text(
        json.dumps([result.__dict__ for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    return results


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "cia" / "results",
    )
    parser.add_argument("--max-parallel-clients", type=int, default=2)
    return parser


def main() -> None:
    args = _parser().parse_args()
    results = run_first_round_cia(
        output_dir=args.output_dir,
        max_parallel_clients=args.max_parallel_clients,
    )
    for result in results:
        print(
            f"{result.privacy:15s} {result.aggregation:10s} "
            f"agg={result.aggregated_test_loss:.3f} "
            f"target={result.target_shadow_loss:.3f} "
            f"diff={result.difference_pct:.3f}%"
        )


if __name__ == "__main__":
    main()
