"""Sweep the privacy x aggregation matrix at 48 clients, using the noise
multiplier chosen from the 8-client noise sweep.

At the paper-default noise_multiplier=0.01, metric-privacy and global-dp
were near-indistinguishable (see ``sweep_noise_multiplier.py``): the
metric-privacy calibration factor (noise_multiplier / max pairwise
client-model distance) was a near-no-op because distance sat right around
1.0. noise_multiplier=0.05 was the sweet spot found there -- training
perturbation at that level pushes client-model distance to ~1.3-1.4,
which lowers metric-privacy's effective noise below global-dp's fixed
noise and produced a real accuracy advantage (+6.9pp homogeneous, +12.2pp
non-iid) at 8 clients, without collapsing training. This sweep checks
whether that advantage holds (or grows/shrinks) at 48 clients.

Scoped to fedavg, fedavgm, and fedyogi (matching the reduced aggregation
list used in the 8-client scaling sweep; fedmedian/fedprox/fedopt dropped
by request there). Reuses ``experiments.reproduce.runner`` unmodified via
subprocess, exactly like ``sweep_8_clients.py`` and
``sweep_noise_multiplier.py``: resumable (skips combinations whose result
JSON already reports the paper-default number of completed rounds),
continues past a failing combination rather than aborting the whole
multi-hour sweep, and supports ``--force`` to ignore existing results and
rerun everything.

48 clients means substantially more per-round work than 8 clients, so
``--max-parallel-clients`` defaults higher here (4 vs. the runner's own
default of 2) to use more of this machine's 12 cores -- tune
``MAX_PARALLEL_CLIENTS`` down if memory pressure becomes an issue (48
clients x more concurrent Ray actors costs more RAM than 8 did).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from metricdp_pytorch.strategy_factory import PRIVACY_MODES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTITION_MODES = ("homogeneous", "non-iid")
AGGREGATION_METHODS_SWEPT = ("fedavg", "fedavgm", "fedyogi")
NUM_CLIENTS = 48
NOISE_MULTIPLIER = 0.05  # chosen from sweep_noise_multiplier.py's 8-client results
EXPECTED_ROUNDS = 20  # paper default (pyproject.toml num-server-rounds)
MAX_PARALLEL_CLIENTS = 4
OUTPUT_DIR = PROJECT_ROOT / "results" / "48client_scaling"
LOG_PATH = OUTPUT_DIR / "sweep_progress.log"


def _log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_name(partition: str, privacy: str, aggregation: str) -> str:
    return f"scaling48__{partition}__{privacy}__{aggregation}"


def result_path(partition: str, privacy: str, aggregation: str) -> Path:
    return OUTPUT_DIR / f"{run_name(partition, privacy, aggregation)}.json"


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


def run_one_combo(
    partition: str,
    privacy: str,
    aggregation: str,
    *,
    force: bool,
    max_parallel_clients: int,
) -> bool:
    """Run one combination; return True on success (or already-complete)."""
    name = run_name(partition, privacy, aggregation)
    path = result_path(partition, privacy, aggregation)
    if not force and is_complete(path):
        _log(f"SKIP  {name} (already complete)")
        return True

    command = [
        sys.executable,
        "-m",
        "experiments.reproduce.runner",
        "--num-clients",
        str(NUM_CLIENTS),
        "--partition",
        partition,
        "--privacy",
        privacy,
        "--aggregation",
        aggregation,
        "--noise-multiplier",
        str(NOISE_MULTIPLIER),
        "--max-parallel-clients",
        str(max_parallel_clients),
        "--output-dir",
        str(OUTPUT_DIR),
        "--run-name",
        name,
    ]
    _log(f"START {name}")
    started = time.monotonic()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
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
    total = len(PARTITION_MODES) * len(PRIVACY_MODES) * len(AGGREGATION_METHODS_SWEPT)
    _log(
        f"Sweep starting: {total} combinations, num_clients={NUM_CLIENTS}, "
        f"noise_multiplier={NOISE_MULTIPLIER}, max_parallel_clients={args.max_parallel_clients}, "
        f"force={args.force}"
    )

    completed = 0
    failed: list[str] = []
    for partition in PARTITION_MODES:
        for privacy in PRIVACY_MODES:
            for aggregation in AGGREGATION_METHODS_SWEPT:
                ok = run_one_combo(
                    partition,
                    privacy,
                    aggregation,
                    force=args.force,
                    max_parallel_clients=args.max_parallel_clients,
                )
                completed += 1
                if not ok:
                    failed.append(run_name(partition, privacy, aggregation))
                _log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")

    _log(f"Sweep finished: {completed}/{total} attempted, {len(failed)} failed")
    if failed:
        _log("Failed combinations: " + ", ".join(failed))


if __name__ == "__main__":
    main()
