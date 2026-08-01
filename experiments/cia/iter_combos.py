"""Shared CIA combo execution iterator."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path

from experiments.reproduce.matrix import Combo
from experiments.reproduce.matrix.run_combo import run_one_combo

LogFunction = Callable[[str], None]


def iter_combos(
    combos: Iterable[Combo],
    *,
    output_dir: Path,
    max_parallel_clients: int,
    log: LogFunction,
    force: bool = False,
) -> Iterator[tuple[bool, Path]]:
    """Run combos sequentially and yield ``(combo, success, model_path)``.

    A model path is yielded even when training fails, allowing callers to
    associate a failure with its combo without reconstructing the filename.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for combo in combos:
        success = run_one_combo(
            combo,
            output_dir=output_dir,
            max_parallel_clients=max_parallel_clients,
            force=force,
            log=log,
            save_model=True,
        )
        yield combo, success, output_dir / f"{combo.run_name()}.pt"
