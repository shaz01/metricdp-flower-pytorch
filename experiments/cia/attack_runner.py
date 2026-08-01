"""Run CIA training combos, evaluate their attack scores, and log progress."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from experiments.cia.datasets.shadow import ShadowDataModule
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
    data_module: ShadowDataModule,
    device: torch.device,
    batch_size: int,
    seed: int,
) -> tuple[float, float, int]:
    model = PaperCNN()
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )

    _validation_loader, test_loader = data_module.server_loaders(
        batch_size=batch_size, seed=seed
    )
    shadow_loader = data_module.target_shadow_loader(
        batch_size=batch_size, seed=seed
    )
    aggregated_metrics = evaluate_model(model, test_loader, device)
    target_metrics = evaluate_model(model, shadow_loader, device)
    return (
        aggregated_metrics["loss"],
        target_metrics["loss"],
        len(shadow_loader.dataset),
    )


def run_attack(
    combos: Sequence[Combo],
    *,
    output_dir: Path,
    log_path: Path,
    max_parallel_clients: int,
    force: bool,
    start_message: str,
    data_module_factory: Callable[[Combo], ShadowDataModule],
    device: torch.device,
    batch_size: int,
    seed: int,
    checkpoint_rounds: tuple[int, ...] = (),
    report_name: str = "cia.json",
) -> list[CiaResult]:
    """Run and evaluate every CIA combo, continuing past failures."""
    if not checkpoint_rounds:
        raise ValueError("CIA attacks require at least one checkpoint round.")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = _logger(log_path)

    total = len(combos)
    log(start_message)
    failed: list[str] = []
    results: list[CiaResult] = []

    for completed, (combo, success, checkpoint_paths) in enumerate(
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
            data_module = data_module_factory(combo)
            for round_number, round_model_path in zip(
                checkpoint_rounds, checkpoint_paths, strict=True
            ):
                aggregated_loss, target_loss, shadow_size = _evaluate_combo(
                    round_model_path,
                    data_module=data_module,
                    device=device,
                    batch_size=batch_size,
                    seed=seed,
                )
                results.append(
                    make_cia_result(
                        combo=combo,
                        server_round=round_number,
                        aggregated_test_loss=aggregated_loss,
                        target_shadow_loss=target_loss,
                        shadow_fraction=data_module.shadow_fraction,
                        shadow_size=shadow_size,
                    )
                )
                log(f"EVALUATED {name} round={round_number}")
        except Exception as error:  # noqa: BLE001 - continue the sweep
            failed.append(name)
            log(f"FAILED {name} (evaluation error: {error})")

        log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")

    report_path = output_dir / report_name
    report_path.write_text(
        json.dumps([result.__dict__ for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"Sweep finished: {total}/{total} attempted, {len(failed)} failed")
    if failed:
        log("Failed combinations: " + ", ".join(failed))
    return results
