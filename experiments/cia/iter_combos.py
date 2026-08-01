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
    checkpoint_rounds: tuple[int, ...] = (),
) -> Iterator[tuple[Combo, bool, tuple[Path, ...]]]:
    """Run combos and yield ``(combo, success, checkpoint_paths)``.

    Checkpoint paths follow the requested round order. The tuple is empty when
    no checkpoint rounds were requested.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for combo in combos:
        success = run_one_combo(
            combo,
            output_dir=output_dir,
            max_parallel_clients=max_parallel_clients,
            force=force,
            log=log,
            checkpoint_rounds=checkpoint_rounds,
        )
        checkpoint_paths = tuple(
            output_dir / f"{combo.run_name()}.round-{round_number}.pt"
            for round_number in checkpoint_rounds
        )
        yield combo, success, checkpoint_paths
