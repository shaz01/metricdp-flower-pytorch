"""Shared sequential runner for client-scaling sweeps."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

from experiments.reproduce.matrix import Combo
from experiments.reproduce.matrix.run_combo import run_one_combo


def run_sweep(
    combos: Sequence[Combo],
    *,
    output_dir: Path,
    log_path: Path,
    max_parallel_clients: int,
    force: bool,
    start_message: str,
) -> None:
    """Run every combo, logging progress while continuing past failures."""
    output_dir.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    total = len(combos)
    log(start_message)

    failed: list[str] = []
    for completed, combo in enumerate(combos, start=1):
        ok = run_one_combo(
            combo,
            output_dir=output_dir,
            max_parallel_clients=max_parallel_clients,
            force=force,
            log=log,
        )
        if not ok:
            failed.append(combo.run_name())
        log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")

    log(f"Sweep finished: {total}/{total} attempted, {len(failed)} failed")
    if failed:
        log("Failed combinations: " + ", ".join(failed))
