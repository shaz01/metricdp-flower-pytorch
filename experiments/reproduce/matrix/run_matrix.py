"""Execution helpers for reproduction matrices."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from experiments.reproduce.matrix.combo import Combo
from experiments.reproduce.matrix.run_combo import run_one_combo


def run_combos(
    combos: list[Combo],
    *,
    output_dir: Path,
    parallel_experiments: int,
    max_parallel_clients: int,
    force: bool,
    client_cpus: float,
) -> None:
    """Execute one parallel pass over the pending matrix runs."""
    progress_path = output_dir / "progress.log"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("a", encoding="utf-8") as progress_file:
        lock = Lock()

        def log(message: str) -> None:
            with lock:
                print(message, flush=True)
                progress_file.write(message + "\n")
                progress_file.flush()

        with ThreadPoolExecutor(max_workers=parallel_experiments) as executor:
            futures = [
                executor.submit(
                    run_one_combo,
                    run,
                    output_dir=output_dir,
                    max_parallel_clients=max_parallel_clients,
                    force=force,
                    log=log,
                    client_cpus=client_cpus,
                    stdout=progress_file,
                )
                for run in combos
            ]
            for future in as_completed(futures):
                future.result()
