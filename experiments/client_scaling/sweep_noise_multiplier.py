"""Sweep the DP noise multiplier at 8 clients to compare metric-privacy vs. global-dp.

Motivation: at the paper-default noise multiplier (0.01), the metric-privacy
wrapper's calibration factor (``noise_multiplier / max pairwise client-model
distance``, see ``metricdp_pytorch.metricdp_strategy``) is a near-no-op --
the logged distances in ``results/8client_scaling/*metric-privacy*`` runs sit
right around 1.0 (0.77-1.46), so metric-privacy's noise ends up almost
identical to global-dp's fixed noise at every round, and both land within
noise of vanilla training. This sweep pushes the noise multiplier up across
a log-ish grid to see whether -- and where -- the two mechanisms actually
diverge, before committing to a value for the 48-client expansion.

Scoped to ``fedavg`` only (not the full aggregation matrix) and both
partition modes, to keep the combinatorics bounded; extend
``AGGREGATION_METHODS_SWEPT`` if a value looks promising and the divergence
should be checked against other aggregators.

Reuses ``experiments.reproduce.runner`` unmodified via subprocess, exactly
like ``sweep_8_clients.py``: resumable (skips combinations whose result JSON
already reports the paper-default number of completed rounds), continues
past a failing combination rather than aborting the whole multi-hour sweep,
and supports ``--force`` to ignore existing results and rerun everything.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTITION_MODES = ("homogeneous", "non-iid")
PRIVACY_MODES_SWEPT = ("global-dp", "metric-privacy")
AGGREGATION_METHODS_SWEPT = ("fedavg",)
NOISE_MULTIPLIERS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
NUM_CLIENTS = 8
EXPECTED_ROUNDS = 20  # paper default (pyproject.toml num-server-rounds)
OUTPUT_DIR = PROJECT_ROOT / "results" / "noise_sweep"
LOG_PATH = OUTPUT_DIR / "sweep_progress.log"


def _log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def format_noise(noise_multiplier: float) -> str:
    """Render a noise multiplier as a filename-safe token, e.g. 0.25 -> '0p25'."""
    return f"{noise_multiplier:g}".replace(".", "p")


def run_name(partition: str, privacy: str, aggregation: str, noise_multiplier: float) -> str:
    return f"noise8__{partition}__{privacy}__{aggregation}__nm{format_noise(noise_multiplier)}"


def result_path(partition: str, privacy: str, aggregation: str, noise_multiplier: float) -> Path:
    return OUTPUT_DIR / f"{run_name(partition, privacy, aggregation, noise_multiplier)}.json"


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
    partition: str, privacy: str, aggregation: str, noise_multiplier: float, *, force: bool
) -> bool:
    """Run one combination; return True on success (or already-complete)."""
    name = run_name(partition, privacy, aggregation, noise_multiplier)
    path = result_path(partition, privacy, aggregation, noise_multiplier)
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
        str(noise_multiplier),
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
    total = (
        len(PARTITION_MODES)
        * len(PRIVACY_MODES_SWEPT)
        * len(AGGREGATION_METHODS_SWEPT)
        * len(NOISE_MULTIPLIERS)
    )
    _log(
        f"Sweep starting: {total} combinations, num_clients={NUM_CLIENTS}, "
        f"noise_multipliers={NOISE_MULTIPLIERS}, force={args.force}"
    )

    completed = 0
    failed: list[str] = []
    for partition in PARTITION_MODES:
        for noise_multiplier in NOISE_MULTIPLIERS:
            for privacy in PRIVACY_MODES_SWEPT:
                for aggregation in AGGREGATION_METHODS_SWEPT:
                    ok = run_one_combo(
                        partition, privacy, aggregation, noise_multiplier, force=args.force
                    )
                    completed += 1
                    if not ok:
                        failed.append(run_name(partition, privacy, aggregation, noise_multiplier))
                    _log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")

    _log(f"Sweep finished: {completed}/{total} attempted, {len(failed)} failed")
    if failed:
        _log("Failed combinations: " + ", ".join(failed))


if __name__ == "__main__":
    main()
