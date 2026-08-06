"""Sweep the DP noise multiplier x client-count matrix to test whether the
8-client sweet spot (``sweep_noise_multiplier.py``, ``nm=0.05``) shifts as
client count grows.

Motivation: ``sweep_48_clients.py`` reused the 8-client sweep's nm=0.05
sweet spot unchanged at 48 clients and found the metric-privacy-vs-global-dp
advantage shrinks/reverses there. That could mean the mechanism itself
degrades with scale -- or it could mean the sweet spot simply moves (e.g. to
a different noise_multiplier) as client count grows, and nm=0.05 was never
the right comparison point at 48 clients to begin with. This sweep extends
``sweep_noise_multiplier.py``'s noise grid into a 2D matrix over client
count to distinguish the two: for each ``num_clients``, sweep the same noise
grid and see where global-dp and metric-privacy actually diverge.

Scoped to ``fedavg`` only and both partition modes, matching
``sweep_noise_multiplier.py``'s own scoping (kept narrow deliberately;
extend ``AGGREGATION_METHODS_SWEPT`` only once a client-count-dependent
sweet spot is confirmed and needs checking against other aggregators). Uses
the fixed paper-default 20-round budget (unlike
``sweep_scale_controlled.py``'s constant-compute control) -- this sweep
answers a different question (does the sweet spot itself shift with n?),
not "is the round budget confounding the comparison?".

Reuses ``experiments.reproduce.runner`` unmodified via subprocess, exactly
like the sibling sweep scripts: resumable (skips combinations whose result
JSON already reports the paper-default number of completed rounds),
continues past a failing combination rather than aborting the whole
multi-hour sweep, and supports ``--force`` to ignore existing results and
rerun everything.

4 client counts x 2 partition modes x 2 privacy modes x 6 noise multipliers
= 96 combinations; at 48 clients in particular this is substantially more
wall-clock time than the 8-client-only ``sweep_noise_multiplier.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.reproduce.matrix import Combo, Hyperparams

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTITION_MODES = ("homogeneous", "non-iid")
PRIVACY_MODES_SWEPT = ("global-dp", "metric-privacy")
AGGREGATION_METHODS_SWEPT = ("fedavg",)
NOISE_MULTIPLIERS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
CLIENT_COUNTS = (8, 16, 32, 48)
EXPECTED_ROUNDS = 20  # paper default (pyproject.toml num-server-rounds)
MAX_PARALLEL_CLIENTS = 4
OUTPUT_DIR = PROJECT_ROOT / "results" / "noise_by_clients"
LOG_PATH = OUTPUT_DIR / "sweep_progress.log"

# Paper-default hyperparameters held fixed across the whole sweep -- same
# values as sweep_noise_multiplier.py / pyproject.toml's
# [tool.flwr.app.config] defaults.
SEED = 42
CLIPPING_NORM = 5.0
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
INITIALIZATION_EPOCHS = 20
DATA_MODULE = "experiments.reproduce.dataset.alzheimer:create_data_module"
MODEL_MODULE = "experiments.reproduce.paper_cnn:create_model"


def _log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def format_noise(noise_multiplier: float) -> str:
    """Render a noise multiplier as a filename-safe token, e.g. 0.25 -> '0p25'."""
    return f"{noise_multiplier:g}".replace(".", "p")


def run_name(
    partition: str, privacy: str, aggregation: str, num_clients: int, noise_multiplier: float
) -> str:
    return (
        f"noisebyclients__{partition}__{privacy}__{aggregation}"
        f"__n{num_clients}__nm{format_noise(noise_multiplier)}"
    )


def result_path(
    partition: str, privacy: str, aggregation: str, num_clients: int, noise_multiplier: float
) -> Path:
    return OUTPUT_DIR / f"{run_name(partition, privacy, aggregation, num_clients, noise_multiplier)}.json"


def is_complete(path: Path, *, expected_rounds: int = EXPECTED_ROUNDS) -> bool:
    """Return whether ``path`` holds a valid, fully-completed result.

    Treats a missing, unparseable, or short-of-rounds file as incomplete, so
    a prior run that was killed mid-write (or mid-sweep) is rerun rather than
    silently accepted.
    """
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    history = data.get("server_evaluate_metrics", {})
    completed_rounds = [int(round_number) for round_number in history if int(round_number) > 0]
    return len(completed_rounds) >= expected_rounds


def runner_args_with_name(
    combo: Combo, *, name: str, max_parallel_clients: int
) -> tuple[str, ...]:
    """Build this combo's runner.py CLI args under this sweep's own run name.

    Delegates to ``Combo.runner_args()`` so every arg runner.py requires
    (--local-epochs/--seed/--clipping-norm/--model-module, missing here
    before this fix -- same bug as the one fixed in
    sweep_scale_controlled.py/sweep_scale_controlled_epochs.py, this script
    just wasn't part of that fix) is included and stays in sync with the
    sibling sweeps. Swaps Combo's trailing ``--run-name`` pair for this
    sweep's own naming (``noisebyclients__...``, no seed/clip/rounds/epochs
    baked in, since those are fixed per sweep here) so
    ``result_path()``/``is_complete()`` keep matching what lands on disk.
    """
    args = combo.runner_args(
        output_dir=OUTPUT_DIR,
        max_parallel_clients=max_parallel_clients,
        client_cpus=1.0,
    )
    assert args[-2] == "--run-name"
    return (*args[:-1], name)


def run_one_combo(
    partition: str,
    privacy: str,
    aggregation: str,
    num_clients: int,
    noise_multiplier: float,
    *,
    force: bool,
    max_parallel_clients: int,
) -> bool:
    """Run one combination; return True on success (or already-complete)."""
    name = run_name(partition, privacy, aggregation, num_clients, noise_multiplier)
    path = result_path(partition, privacy, aggregation, num_clients, noise_multiplier)
    if not force and is_complete(path):
        _log(f"SKIP  {name} (already complete)")
        return True

    combo = Combo(
        name_prefix="noisebyclients",
        num_clients=num_clients,
        partition=partition,
        privacy=privacy,
        aggregation=aggregation,
        seed=SEED,
        noise_multiplier=noise_multiplier,
        hyperparams=Hyperparams(
            clipping_norm=CLIPPING_NORM,
            rounds=EXPECTED_ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            initialization_epochs=INITIALIZATION_EPOCHS,
        ),
        data_module=DATA_MODULE,
        model_module=MODEL_MODULE,
    )
    command = [
        sys.executable,
        "-m",
        "experiments.reproduce.runner",
        *runner_args_with_name(
            combo, name=name, max_parallel_clients=max_parallel_clients
        ),
    ]
    _log(f"START {name}")
    started = time.monotonic()
    child_env = os.environ.copy()
    # Same determinism env vars experiments/reproduce/matrix/run_combo.py
    # sets for the sibling sweeps.
    child_env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    child_env.setdefault("PYTHONHASHSEED", "0")
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=child_env)
    elapsed = time.monotonic() - started
    if result.returncode == 0:
        _log(f"DONE  {name} ({elapsed:.1f}s)")
        return True
    _log(f"FAILED {name} (exit={result.returncode}, {elapsed:.1f}s)")
    return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun every combination even if a complete result already exists",
    )
    parser.add_argument(
        "--max-parallel-clients",
        type=int,
        default=MAX_PARALLEL_CLIENTS,
        help="cap simultaneous Ray actors to control memory use",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = (
        len(CLIENT_COUNTS)
        * len(PARTITION_MODES)
        * len(PRIVACY_MODES_SWEPT)
        * len(AGGREGATION_METHODS_SWEPT)
        * len(NOISE_MULTIPLIERS)
    )
    _log(
        f"Sweep starting: {total} combinations, client_counts={CLIENT_COUNTS}, "
        f"noise_multipliers={NOISE_MULTIPLIERS}, "
        f"max_parallel_clients={args.max_parallel_clients}, force={args.force}"
    )

    completed = 0
    failed: list[str] = []
    for num_clients in CLIENT_COUNTS:
        for partition in PARTITION_MODES:
            for noise_multiplier in NOISE_MULTIPLIERS:
                for privacy in PRIVACY_MODES_SWEPT:
                    for aggregation in AGGREGATION_METHODS_SWEPT:
                        ok = run_one_combo(
                            partition,
                            privacy,
                            aggregation,
                            num_clients,
                            noise_multiplier,
                            force=args.force,
                            max_parallel_clients=args.max_parallel_clients,
                        )
                        completed += 1
                        if not ok:
                            failed.append(
                                run_name(
                                    partition, privacy, aggregation, num_clients, noise_multiplier
                                )
                            )
                        _log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")

    _log(f"Sweep finished: {completed}/{total} attempted, {len(failed)} failed")
    if failed:
        _log("Failed combinations: " + ", ".join(failed))


if __name__ == "__main__":
    main()
