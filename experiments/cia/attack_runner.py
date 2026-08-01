"""Run CIA training combos, evaluate their attack scores, and log progress."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from experiments.cia.datasets.paper import PaperShadowDataModule
from experiments.cia.iter_combos import iter_combos
from experiments.cia.result import CiaResult, make_cia_result
from experiments.reproduce.matrix import Combo
from experiments.reproduce.paper_cnn import PaperCNN
from experiments.reproduce.paper_loss import evaluate_model


def _logger(log_path: Path):
    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def _evaluate_combo(
    model_path: Path,
    *,
    data_module: PaperShadowDataModule,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[float, float]:
    model = PaperCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    _validation_loader, test_loader = data_module.server_loaders(
        batch_size=batch_size, seed=seed
    )
    shadow_loader = data_module.target_shadow_loader(
        batch_size=batch_size, seed=seed
    )
    aggregated_metrics = evaluate_model(model, test_loader, device)
    target_metrics = evaluate_model(model, shadow_loader, device)
    return aggregated_metrics["loss"], target_metrics["loss"]


def run_attack(
    combos: Sequence[Combo],
    *,
    output_dir: Path,
    log_path: Path,
    max_parallel_clients: int,
    force: bool,
    start_message: str,
    data_module: PaperShadowDataModule,
    device: torch.device,
    batch_size: int,
    seed: int,
    checkpoint_rounds: tuple[int, ...] = (),
) -> list[CiaResult]:
    """Run and evaluate every CIA combo, continuing past failures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = _logger(log_path)

    total = len(combos)
    log(start_message)
    failed: list[str] = []
    results: list[CiaResult] = []

    for completed, (combo, success, model_path) in enumerate(
        iter_combos(
            combos,
            output_dir=output_dir,
            max_parallel_clients=max_parallel_clients,
            force=force,
            log=log,
            checkpoint_rounds=checkpoint_rounds,
        ),
        start=1,
    ):
        name = combo.run_name()
        if not success:
            failed.append(name)
            log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")
            continue

        try:
            round_models = (
                tuple(
                    (
                        round_number,
                        output_dir / f"{name}.round-{round_number}.pt",
                    )
                    for round_number in checkpoint_rounds
                )
                or ((combo.hyperparams.rounds, model_path),)
            )
            for round_number, round_model_path in round_models:
                aggregated_loss, target_loss = _evaluate_combo(
                    round_model_path,
                    data_module=data_module,
                    device=device,
                    batch_size=batch_size,
                    seed=seed,
                )
                results.append(
                    make_cia_result(
                        privacy=combo.privacy,
                        aggregation=combo.aggregation,
                        server_round=round_number,
                        aggregated_test_loss=aggregated_loss,
                        target_shadow_loss=target_loss,
                    )
                )
                log(f"EVALUATED {name} round={round_number}")
        except Exception as error:  # noqa: BLE001 - continue the sweep
            failed.append(name)
            log(f"FAILED {name} (evaluation error: {error})")

        log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")

    report_path = output_dir / "first_round_cia.json"
    report_path.write_text(
        json.dumps([result.__dict__ for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"Sweep finished: {total}/{total} attempted, {len(failed)} failed")
    if failed:
        log("Failed combinations: " + ", ".join(failed))
    return results
