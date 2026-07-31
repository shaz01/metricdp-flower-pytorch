"""Sweep the privacy x aggregation matrix across client counts under a
constant-compute control, instead of the fixed 20-round budget every prior
sweep (``sweep_8_clients.py``, ``sweep_48_clients.py``) used regardless of
client count.

Motivation: with the paper's homogeneous partitioning, each client's shard
shrinks roughly as ``1 / num_clients`` (and non-iid quantity-skew scales the
same way on average). At a fixed 20-round, 5-local-epoch budget, a client at
48 clients therefore trains on ~6x less data per round than a client at 8
clients -- so ``results/48client_scaling``'s shrinking metric-privacy
advantage could be a genuine mechanism failure at scale, or it could just be
this round-budget confound (less total gradient signal per client, nothing
to do with the privacy mechanism). This sweep disentangles the two by
scaling ``--rounds`` with ``num_clients`` so each client's total gradient
steps over its own data stays roughly constant across client counts:

    rounds(n) = round(BASE_ROUNDS * n / BASE_NUM_CLIENTS)

with ``--local-epochs``/``--batch-size`` held fixed. This is a proportional
approximation (average shard size, not per-client exact accounting), which
is the same fidelity the existing quantity-skewed non-iid partitioning
already operates at.

Scoped to fedavg and fedyogi (matching the reduced aggregation list used in
the 8-client/48-client scaling sweeps) at noise_multiplier=0.05, the sweet
spot from ``sweep_noise_multiplier.py``'s 8-client sweep. Reuses
``experiments.reproduce.runner`` unmodified via subprocess, exactly like the
sibling sweep scripts: resumable (skips combinations whose result JSON
already reports that client count's scaled round total as completed),
continues past a failing combination rather than aborting the whole
multi-hour sweep, and supports ``--force`` to ignore existing results and
rerun everything.

Client counts above the base scale to substantially more rounds (48 clients
-> 12x the base rounds), so this sweep costs much more wall-clock time than
``sweep_48_clients.py``'s fixed-round version -- expect a multi-hour-to-
multi-day run at the higher client counts.
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
AGGREGATION_METHODS_SWEPT = ("fedavg", "fedyogi")
CLIENT_COUNTS = (4, 8, 16, 48)
BASE_NUM_CLIENTS = 4
BASE_ROUNDS = 20  # paper default (pyproject.toml num-server-rounds)
NOISE_MULTIPLIER = 0.05  # chosen from sweep_noise_multiplier.py's 8-client results
MAX_PARALLEL_CLIENTS = 4
OUTPUT_DIR = PROJECT_ROOT / "results" / "scale_controlled"
LOG_PATH = OUTPUT_DIR / "sweep_progress.log"


def rounds_for(num_clients: int) -> int:
    """Scale rounds so per-client total gradient steps stay ~constant.

    Homogeneous (and, on average, quantity-skewed non-iid) partitioning
    splits a fixed dataset across ``num_clients`` shards, so each client's
    shard size is roughly proportional to ``1 / num_clients``. Multiplying
    rounds by ``num_clients / BASE_NUM_CLIENTS`` compensates for that
    shrinkage, holding roughly constant the total number of gradient steps
    each client takes over its own data across the whole run.
    """
    return round(BASE_ROUNDS * num_clients / BASE_NUM_CLIENTS)


def _log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_name(partition: str, privacy: str, aggregation: str, num_clients: int) -> str:
    return f"scalectrl__{partition}__{privacy}__{aggregation}__n{num_clients}"


def result_path(partition: str, privacy: str, aggregation: str, num_clients: int) -> Path:
    return OUTPUT_DIR / f"{run_name(partition, privacy, aggregation, num_clients)}.json"


def is_complete(path: Path, *, expected_rounds: int) -> bool:
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
    num_clients: int,
    *,
    force: bool,
    max_parallel_clients: int,
) -> bool:
    """Run one combination; return True on success (or already-complete)."""
    name = run_name(partition, privacy, aggregation, num_clients)
    path = result_path(partition, privacy, aggregation, num_clients)
    rounds = rounds_for(num_clients)
    if not force and is_complete(path, expected_rounds=rounds):
        _log(f"SKIP  {name} (already complete)")
        return True

    command = [
        sys.executable,
        "-m",
        "experiments.reproduce.runner",
        "--num-clients",
        str(num_clients),
        "--partition",
        partition,
        "--privacy",
        privacy,
        "--aggregation",
        aggregation,
        "--noise-multiplier",
        str(NOISE_MULTIPLIER),
        "--rounds",
        str(rounds),
        "--max-parallel-clients",
        str(max_parallel_clients),
        "--output-dir",
        str(OUTPUT_DIR),
        "--run-name",
        name,
    ]
    _log(f"START {name} (rounds={rounds})")
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
    total = (
        len(CLIENT_COUNTS)
        * len(PARTITION_MODES)
        * len(PRIVACY_MODES)
        * len(AGGREGATION_METHODS_SWEPT)
    )
    _log(
        f"Sweep starting: {total} combinations, client_counts={CLIENT_COUNTS}, "
        f"rounds={[rounds_for(n) for n in CLIENT_COUNTS]}, "
        f"noise_multiplier={NOISE_MULTIPLIER}, "
        f"max_parallel_clients={args.max_parallel_clients}, force={args.force}"
    )

    completed = 0
    failed: list[str] = []
    for num_clients in CLIENT_COUNTS:
        for partition in PARTITION_MODES:
            for privacy in PRIVACY_MODES:
                for aggregation in AGGREGATION_METHODS_SWEPT:
                    ok = run_one_combo(
                        partition,
                        privacy,
                        aggregation,
                        num_clients,
                        force=args.force,
                        max_parallel_clients=args.max_parallel_clients,
                    )
                    completed += 1
                    if not ok:
                        failed.append(run_name(partition, privacy, aggregation, num_clients))
                    _log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")

    _log(f"Sweep finished: {completed}/{total} attempted, {len(failed)} failed")
    if failed:
        _log("Failed combinations: " + ", ".join(failed))


if __name__ == "__main__":
    main()
