"""Run CIA training combos, evaluate their attack scores, and log progress."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score

from experiments.cia.datasets.shadow import ShadowDataModule
from experiments.cia.iter_combos import iter_combos
from experiments.cia.result import CiaResult, make_cia_result
from experiments.reproduce.matrix import Combo
from experiments.reproduce.paper_loss import (
    evaluate_model,
    sparse_categorical_cross_entropy,
)
from metricdp_pytorch.model_module import load_model


def _logger(log_path: Path):
    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log


def _membership_scores(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> list[float]:
    """Return per-example membership scores ``-loss(W, x)``."""
    scores: list[float] = []
    model.to(device)
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            probabilities = model(images.to(device, non_blocking=True))
            losses = sparse_categorical_cross_entropy(
                probabilities,
                labels.to(device, non_blocking=True),
                reduction="none",
            )
            scores.extend((-losses).cpu().tolist())
    return [float(score) for score in scores]


def _evaluate_combo(
    model_path: Path,
    *,
    data_module: ShadowDataModule,
    device: torch.device,
    combo: Combo,
) -> tuple[float, float, int, list[float], list[float], float]:
    model = load_model(combo.model_module)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )

    _validation_loader, test_loader = data_module.server_loaders(
        batch_size=combo.hyperparams.batch_size, seed=combo.seed
    )
    shadow_loader = data_module.target_shadow_loader(
        batch_size=combo.hyperparams.batch_size, seed=combo.seed
    )
    aggregated_metrics = evaluate_model(model, test_loader, device)
    target_metrics = evaluate_model(model, shadow_loader, device)
    in_scores = _membership_scores(model, shadow_loader, device)
    out_scores = _membership_scores(model, test_loader, device)
    attack_auc = float(
        roc_auc_score(
            [1] * len(in_scores) + [0] * len(out_scores),
            in_scores + out_scores,
        )
    )
    return (
        aggregated_metrics["loss"],
        target_metrics["loss"],
        len(shadow_loader.dataset),
        in_scores,
        out_scores,
        attack_auc,
    )


def _write_report(report_path: Path, results: Sequence[CiaResult]) -> None:
    """Atomically persist CIA results so an interrupted sweep can resume."""
    temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps([result.__dict__ for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(report_path)


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
    report_path = output_dir / report_name
    results: list[CiaResult] = []
    if report_path.exists():
        saved_results = json.loads(report_path.read_text(encoding="utf-8"))
        results = [
            CiaResult(**saved_result)
            for saved_result in saved_results
            if "run_name" in saved_result and "seed" in saved_result
        ]

    completed_keys = {
        (result.run_name, result.server_round) for result in results
    }
    pending_combos = [
        combo
        for combo in combos
        if force
        or not all(
            (combo.run_name(), round_number) in completed_keys
            for round_number in checkpoint_rounds
        )
    ]
    skipped = total - len(pending_combos)
    if skipped:
        log(f"SKIP  {skipped} combinations (attack results already exist)")

    failed: list[str] = []
    for completed, (combo, success, checkpoint_paths) in enumerate(
        iter_combos(
            pending_combos,
            output_dir=output_dir,
            max_parallel_clients=max_parallel_clients,
            force=force,
            log=log,
            checkpoint_rounds=checkpoint_rounds,
        ),
        start=skipped + 1,
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
                (
                    aggregated_loss,
                    target_loss,
                    shadow_size,
                    in_scores,
                    out_scores,
                    attack_auc,
                ) = _evaluate_combo(
                    round_model_path,
                    data_module=data_module,
                    device=device,
                    combo=combo,
                )
                result = make_cia_result(
                    combo=combo,
                    server_round=round_number,
                    aggregated_test_loss=aggregated_loss,
                    target_shadow_loss=target_loss,
                    shadow_fraction=data_module.shadow_fraction,
                    shadow_size=shadow_size,
                    in_scores=in_scores,
                    out_scores=out_scores,
                    attack_auc=attack_auc,
                )
                result_key = (result.run_name, result.server_round)
                results = [
                    existing
                    for existing in results
                    if (existing.run_name, existing.server_round) != result_key
                ]
                results.append(result)
                _write_report(report_path, results)
                round_model_path.unlink()
                log(f"EVALUATED {name} round={round_number} (checkpoint deleted)")
        except Exception as error:  # noqa: BLE001 - continue the sweep
            failed.append(name)
            log(f"FAILED {name} (evaluation error: {error})")

        log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")

    _write_report(report_path, results)
    log(f"Sweep finished: {total}/{total} attempted, {len(failed)} failed")
    if failed:
        log("Failed combinations: " + ", ".join(failed))
    return results
