"""Shared execution helper for reproduction matrix combinations."""

from __future__ import annotations

import json
import os
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
    checkpoint_rounds: tuple[int, ...] = (),
) -> bool:
    """Run one combo through the reproduction runner, or skip it when complete."""
    if any(value < 1 or value > combo.hyperparams.rounds for value in checkpoint_rounds):
        raise ValueError("checkpoint_rounds must be between 1 and combo rounds.")
    name = combo.run_name()
    result_path = combo.result_path(output_dir)

    required_model_paths = [
        output_dir / f"{name}.round-{round_number}.pt"
        for round_number in checkpoint_rounds
    ]
    if (
        not force
        and is_complete(result_path, expected_rounds=combo.hyperparams.rounds)
        and all(path.exists() for path in required_model_paths)
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
            checkpoint_rounds=checkpoint_rounds,
        ),
    ]
    log(f"START {name} {start_detail}")

    started = time.monotonic()
    subprocess_options: dict[str, object] = {}
    if stdout is not None:
        subprocess_options["stdout"] = stdout
        subprocess_options["stderr"] = subprocess.STDOUT
    child_env = os.environ.copy()
    if env is not None:
        child_env.update(env)
    child_env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    child_env.setdefault("PYTHONHASHSEED", "0")
    subprocess_options["env"] = child_env
    result = subprocess.run(command, cwd=PROJECT_ROOT, **subprocess_options)
    elapsed = time.monotonic() - started

    if result.returncode == 0:
        log(f"DONE  {name} ({elapsed:.1f}s)")
        return True
    log(f"FAILED {name} (exit={result.returncode}, {elapsed:.1f}s)")
    return False
