"""Sweep the paper's full privacy x aggregation matrix at 8 clients instead of 4.

Reuses ``experiments.reproduce.runner`` unmodified via subprocess, once per
(partition_mode, privacy, aggregation) combination, with paper-default
rounds/local-epochs/batch-size/noise settings and ``--num-clients 8``. This
is a scaling check, not a paper-numbers reproduction: the paper's own tables
are all reported for 4 clients.

Continues past a failing combination (logging it) rather than aborting the
whole multi-hour sweep, since a single combo failure shouldn't lose the rest
of an unattended run.

Resumable: before launching a combination, checks whether its result JSON
already exists and reports the paper-default number of completed rounds; if
so, the combination is skipped. This lets an interrupted or killed sweep
(e.g. restarted after a code change) pick back up without redoing already
-finished combinations. Pass ``--force`` to ignore existing results and rerun
everything.
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
# fedmedian, fedprox, and fedopt were dropped from this sweep by request.
SWEEP_AGGREGATION_METHODS = ("fedavg", "fedavgm", "fedyogi")
NUM_CLIENTS = 8
EXPECTED_ROUNDS = 20  # paper default (pyproject.toml num-server-rounds); sweep doesn't override --rounds
OUTPUT_DIR = PROJECT_ROOT / "results" / "8client_scaling"
LOG_PATH = OUTPUT_DIR / "sweep_progress.log"


def _log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_name(partition: str, privacy: str, aggregation: str) -> str:
    return f"scaling8__{partition}__{privacy}__{aggregation}"


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


def run_one_combo(partition: str, privacy: str, aggregation: str, *, force: bool) -> bool:
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
    return parser


def main() -> None:
    args = _parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = len(PARTITION_MODES) * len(PRIVACY_MODES) * len(SWEEP_AGGREGATION_METHODS)
    _log(f"Sweep starting: {total} combinations, num_clients={NUM_CLIENTS}, force={args.force}")

    completed = 0
    failed: list[str] = []
    for partition in PARTITION_MODES:
        for privacy in PRIVACY_MODES:
            for aggregation in SWEEP_AGGREGATION_METHODS:
                ok = run_one_combo(partition, privacy, aggregation, force=args.force)
                completed += 1
                if not ok:
                    failed.append(run_name(partition, privacy, aggregation))
                _log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")

    _log(f"Sweep finished: {completed}/{total} attempted, {len(failed)} failed")
    if failed:
        _log("Failed combinations: " + ", ".join(failed))


if __name__ == "__main__":
    main()
