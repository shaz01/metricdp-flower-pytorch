"""Run CIA training combos, evaluate their attack scores, and log progress."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import torch

from experiments.cia import cia
from experiments.cia.datasets.shadow import ShadowDataModule
from experiments.cia.iter_combos import iter_combos
from experiments.cia.result import CiaResult, make_cia_result
from experiments.reproduce.matrix import Combo
from metricdp_pytorch.utils.logging import make_file_logger


def _load_results(report_path: Path) -> list[CiaResult]:
    """Load resumable CIA results from an existing report."""
    if not report_path.exists():
        return []
    saved_results = json.loads(report_path.read_text(encoding="utf-8"))
    required_loss_fields = {
        "target_clean_shadow_loss",
        "target_noisy_shadow_loss",
    }
    return [
        CiaResult(**saved_result)
        for saved_result in saved_results
        if "run_name" in saved_result
        and "seed" in saved_result
        and required_loss_fields <= saved_result.keys()
    ]


def _pending_combos(
        combos: Sequence[Combo],
        results: Sequence[CiaResult],
        checkpoint_rounds: tuple[int, ...],
        *,
        force: bool,
) -> list[Combo]:
    """Return combos with at least one checkpoint result still missing."""
    completed_keys = {
        (result.run_name, result.server_round) for result in results
    }
    return [
        combo
        for combo in combos
        if force
           or not all(
            (combo.run_name(), round_number) in completed_keys
            for round_number in checkpoint_rounds
        )
    ]


def _write_report(report_path: Path, results: Sequence[CiaResult]) -> None:
    """Atomically persist CIA results so an interrupted sweep can resume."""
    temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps([result.__dict__ for result in results], indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(report_path)


def _replace_result(results: list[CiaResult], result: CiaResult) -> None:
    """Replace the result for the same run and round, or append it."""
    result_key = (result.run_name, result.server_round)
    results[:] = [
        existing
        for existing in results
        if (existing.run_name, existing.server_round) != result_key
    ]
    results.append(result)


def _evaluate_checkpoints(
        combo: Combo,
        checkpoint_rounds: tuple[int, ...],
        checkpoint_paths: Sequence[Path],
        *,
        clean_data_module_factory: Callable[[Combo], ShadowDataModule],
        noisy_data_module_factory: Callable[[Combo], ShadowDataModule],
        device: torch.device,
        report_path: Path,
        results: list[CiaResult],
        log: Callable[[str], None],
) -> None:
    """Evaluate, persist, and delete every checkpoint for one combo."""
    clean_data_module = clean_data_module_factory(combo)
    noisy_data_module = noisy_data_module_factory(combo)
    if clean_data_module.shadow_fraction != noisy_data_module.shadow_fraction:
        raise ValueError("Clean and noisy shadow fractions must match.")
    name = combo.run_name()

    # For each specified checkpoint round, evaluate the model and persist the result.
    for round_number, round_model_path in zip(
            checkpoint_rounds, checkpoint_paths, strict=True
    ):
        (
            aggregated_loss,
            clean_target_loss,
            noisy_target_loss,
            shadow_size,
        ) = cia.eval_model(
            round_model_path,
            clean_data_module=clean_data_module,
            noisy_data_module=noisy_data_module,
            device=device,
            combo=combo,
        )
        result = make_cia_result(
            combo=combo,
            server_round=round_number,
            aggregated_test_loss=aggregated_loss,
            target_clean_shadow_loss=clean_target_loss,
            target_noisy_shadow_loss=noisy_target_loss,
            shadow_fraction=clean_data_module.shadow_fraction,
            shadow_size=shadow_size,
        )
        _replace_result(results, result)
        _write_report(report_path, results)
        # Delete the checkpoint after evaluation
        round_model_path.unlink()

        log(f"EVALUATED {name} round={round_number} (checkpoint deleted)")


def run_attack(
        combos: Sequence[Combo],
        *,
        output_dir: Path,
        log_path: Path,
        max_parallel_clients: int,
        force: bool,
        start_message: str,
        clean_data_module_factory: Callable[[Combo], ShadowDataModule],
        noisy_data_module_factory: Callable[[Combo], ShadowDataModule],
        device: torch.device,
        checkpoint_rounds: tuple[int, ...] = (),
        report_name: str = "cia.json",
) -> list[CiaResult]:
    """Run and evaluate every CIA combo, continuing past failures."""
    if not checkpoint_rounds:
        raise ValueError("CIA attacks require at least one checkpoint round.")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = make_file_logger(log_path)

    total = len(combos)
    log(start_message)
    report_path = output_dir / report_name
    results = _load_results(report_path)

    pending_combos = _pending_combos(
        combos, results, checkpoint_rounds, force=force
    )
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
            _evaluate_checkpoints(
                combo,
                checkpoint_rounds,
                checkpoint_paths,
                clean_data_module_factory=clean_data_module_factory,
                noisy_data_module_factory=noisy_data_module_factory,
                device=device,
                report_path=report_path,
                results=results,
                log=log,
            )
        except Exception as error:  # noqa: BLE001 - continue the sweep
            failed.append(name)
            log(f"FAILED {name} (evaluation error: {error})")

        log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")

    _write_report(report_path, results)
    log(f"Sweep finished: {total}/{total} attempted, {len(failed)} failed")
    if failed:
        log("Failed combinations: " + ", ".join(failed))
    return results
