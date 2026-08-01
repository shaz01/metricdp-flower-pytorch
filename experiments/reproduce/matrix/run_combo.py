"""Shared execution helper for reproduction matrix combinations."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TextIO

from experiments.reproduce.matrix.combo import Combo

PROJECT_ROOT = Path(__file__).resolve().parents[3]

LogFunction = Callable[[str], None]


def is_complete(path: Path, *, expected_rounds: int) -> bool:
    """Return whether a result contains the expected completed server rounds."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        history = data.get("server_evaluate_metrics", {})
        completed_rounds = [
            int(round_number) for round_number in history if int(round_number) > 0
        ]
    except (AttributeError, json.JSONDecodeError, OSError, TypeError, ValueError):
        return False
    return len(completed_rounds) >= expected_rounds


def run_one_combo(
    combo: Combo,
    *,
    output_dir: Path,
    max_parallel_clients: int,
    force: bool,
    client_cpus: float = 1.0,

    log: LogFunction,
    env: Mapping[str, str] | None = None,
    start_detail: str = "",
    stdout: TextIO | None = None,
    save_model: bool = False,
) -> bool:
    """Run one combo through the reproduction runner, or skip it when complete."""
    name = combo.run_name()
    result_path = combo.result_path(output_dir)

    model_path = output_dir / f"{name}.pt"
    if (
        not force
        and is_complete(result_path, expected_rounds=combo.hyperparams.rounds)
        and (not save_model or model_path.exists())
    ):
        log(f"SKIP  {name} (already complete)")
        return True

    command = [
        sys.executable,
        "-m",
        "experiments.reproduce.runner",
        *combo.runner_args(
            output_dir=output_dir,
            max_parallel_clients=max_parallel_clients,
            client_cpus=client_cpus,
            save_model=save_model,
        ),
    ]
    log(f"START {name} {start_detail}")

    started = time.monotonic()
    subprocess_options: dict[str, object] = {}
    if stdout is not None:
        subprocess_options["stdout"] = stdout
        subprocess_options["stderr"] = subprocess.STDOUT
    if env is not None:
        subprocess_options["env"] = dict(env)
    result = subprocess.run(command, cwd=PROJECT_ROOT, **subprocess_options)
    elapsed = time.monotonic() - started

    if result.returncode == 0:
        log(f"DONE  {name} ({elapsed:.1f}s)")
        return True
    log(f"FAILED {name} (exit={result.returncode}, {elapsed:.1f}s)")
    return False
