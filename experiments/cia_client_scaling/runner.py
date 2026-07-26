"""Orchestrate the 48-client Client Inference Attack experiment.

For each of 12 ``(partition_mode, privacy, aggregation)`` combinations, run
two timing variants:

- ``first-round``: mirrors ``experiments/cia/runner.py``'s paper-exact
  methodology (1 round, local-epochs=20), scaled from 3 to 48 clients.
- ``post-convergence``: mirrors ``experiments/client_scaling/
  sweep_48_clients.py``'s actual training regime (20 rounds,
  local-epochs=5, noise-multiplier=0.05), with ``--save-model`` added.

Both shell out to the existing, unmodified ``experiments.reproduce.runner``
CLI with the default Alzheimer data module, then evaluate the resulting
saved model's loss on the global test set and on a fixed target client's
(``partition_id=0``) shadow split, reporting the relative-difference attack
score for each combination.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NUM_CLIENTS = 48
TARGET_PARTITION_ID = 0
SEED = 42
BATCH_SIZE = 32
PARTITION_MODES = ("homogeneous", "non-iid")
AGGREGATIONS = ("fedavg", "fedyogi")

TIMING_CONFIGS: dict[str, dict[str, int | float]] = {
    "first-round": {
        "rounds": 1,
        "local_epochs": 20,
        "noise_multiplier": 0.01,
        "clipping_norm": 5.0,
    },
    "post-convergence": {
        "rounds": 20,
        "local_epochs": 5,
        "noise_multiplier": 0.05,
        "clipping_norm": 5.0,
    },
}
TIMINGS = tuple(TIMING_CONFIGS)


def run_name(partition_mode: str, timing: str, privacy: str, aggregation: str) -> str:
    return f"cia_scaling__{timing}__{partition_mode}__{privacy}__{aggregation}"


def build_reproduce_command(
    *,
    partition_mode: str,
    timing: str,
    privacy: str,
    aggregation: str,
    output_dir: Path,
    max_parallel_clients: int,
) -> list[str]:
    """Build the argv for one real 48-client CIA training run."""
    timing_config = TIMING_CONFIGS[timing]
    name = run_name(partition_mode, timing, privacy, aggregation)
    return [
        sys.executable,
        "-m",
        "experiments.reproduce.runner",
        "--num-clients",
        str(NUM_CLIENTS),
        "--partition",
        partition_mode,
        "--privacy",
        privacy,
        "--aggregation",
        aggregation,
        "--rounds",
        str(timing_config["rounds"]),
        "--local-epochs",
        str(timing_config["local_epochs"]),
        "--noise-multiplier",
        str(timing_config["noise_multiplier"]),
        "--clipping-norm",
        str(timing_config["clipping_norm"]),
        "--seed",
        str(SEED),
        "--output-dir",
        str(output_dir),
        "--run-name",
        name,
        "--save-model",
        "--max-parallel-clients",
        str(max_parallel_clients),
    ]


def is_training_complete(path: Path, *, expected_rounds: int) -> bool:
    """Return whether ``path`` holds a valid, fully-completed training result.

    Treats a missing, unparseable, or short-of-rounds file as incomplete, so
    a prior run that was killed mid-write (or mid-sweep) is rerun rather than
    silently accepted. Mirrors
    ``experiments/client_scaling/sweep_48_clients.py``'s ``is_complete``.
    """
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    history = data.get("server_evaluate_metrics", {})
    completed_rounds = [
        int(round_number) for round_number in history if int(round_number) > 0
    ]
    return len(completed_rounds) >= expected_rounds
