"""Orchestrate the paper's first-round single-shot CIA experiment (Section 7.4.1).

For each of the 18 (privacy, aggregation) combinations, this launches one
real 1-round, 3-client Flower simulation by shelling out to the existing,
unmodified ``experiments.reproduce.runner`` CLI (pointed at this package's
paper-exact shadow data-module factory), then evaluates the resulting saved
model's loss on the global test set and on the target client's shadow split,
reporting
the relative-difference attack score for each combination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from experiments.cia.result import CiaResult, make_cia_result
from experiments.cia.datasets.paper import (
    PAPER_CIA_NUM_CLIENTS,
    PaperShadowDataModule,
)
from experiments.cia.iter_combos import iter_combos
from experiments.reproduce.matrix import Combo, Hyperparams, Matrix
from experiments.reproduce.paper_cnn import PaperCNN
from experiments.reproduce.paper_loss import evaluate_model
from metricdp_pytorch.strategy_factory import AGGREGATION_METHODS, PRIVACY_MODES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CIA_LOCAL_EPOCHS = 20
CIA_SEED = 42
CIA_NOISE_MULTIPLIER = 0.01
CIA_CLIPPING_NORM = 5.0
CIA_BATCH_SIZE = 32

CIA_MATRIX = Matrix(
    partitions=("homogeneous",),
    privacy_modes=tuple(PRIVACY_MODES),
    aggregations=tuple(AGGREGATION_METHODS),
    seeds=(CIA_SEED,),
    noise_multipliers=(CIA_NOISE_MULTIPLIER,),
    hyperparams=Hyperparams(
        clipping_norm=CIA_CLIPPING_NORM,
        rounds=1,
        local_epochs=CIA_LOCAL_EPOCHS,
        batch_size=CIA_BATCH_SIZE,
        learning_rate=0.001,
        initialization_epochs=20,
    ),
    data_module="experiments.cia.datasets.paper:create_paper_shadow_data_module",
)


def build_cia_combos(
    *,
    privacy_modes: tuple[str, ...] = tuple(PRIVACY_MODES),
    aggregations: tuple[str, ...] = tuple(AGGREGATION_METHODS),
) -> list[Combo]:
    """Return the requested paper privacy × aggregation matrix."""
    matrix = Matrix(
        partitions=CIA_MATRIX.partitions,
        privacy_modes=privacy_modes,
        aggregations=aggregations,
        seeds=CIA_MATRIX.seeds,
        noise_multipliers=CIA_MATRIX.noise_multipliers,
        hyperparams=CIA_MATRIX.hyperparams,
        data_module=CIA_MATRIX.data_module,
    )
    return matrix.list_combos(
        name_prefix="cia", num_clients=PAPER_CIA_NUM_CLIENTS
    )


def evaluate_combo(
    model_path: Path,
    *,
    data_module: PaperShadowDataModule,
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
    force: bool = False,
) -> list[CiaResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_module = PaperShadowDataModule()

    results: list[CiaResult] = []
    combos = build_cia_combos(
        privacy_modes=privacy_modes, aggregations=aggregations
    )
    for combo, success, model_path in iter_combos(
        combos,
        output_dir=output_dir,
        max_parallel_clients=max_parallel_clients,
        force=force,
        log=lambda message: print(message, flush=True),
    ):
        if not success:
            continue
        aggregated_loss, target_loss = evaluate_combo(
            model_path, data_module=data_module, device=device
        )
        results.append(
            make_cia_result(
                privacy=combo.privacy,
                aggregation=combo.aggregation,
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun combinations even when their result and model already exist",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    results = run_first_round_cia(
        output_dir=args.output_dir,
        max_parallel_clients=args.max_parallel_clients,
        force=args.force,
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
