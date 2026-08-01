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

Scoped to the minimum matrix that answers the confound-vs-real-failure
question, not the full 12-combo grid: only ``global-dp`` and
``metric-privacy`` (``vanilla`` isn't part of the gap being measured) on
``fedavg`` only (``fedyogi`` can follow once there's a signal), at
noise_multiplier=0.05, the sweet spot from ``sweep_noise_multiplier.py``'s
8-client sweep. ``CLIENT_COUNTS`` is ``{4, 8, 48}`` rather than the full
``{4, 8, 16, 48}`` -- n=4 is the free/already-complete anchor, n=8 is a cheap
sanity check that the compute-scaling methodology itself doesn't introduce
artifacts, and n=48 is the decisive test (does the gap that shrinks at
``results/48client_scaling``'s fixed 20 rounds come back at the
compute-matched 240 rounds). n=16 is an interpolation point for the
eventual paper's dose-response curve, not needed for that yes/no answer, so
it's deferred -- rerun with a wider ``CLIENT_COUNTS`` later once the n=48
result direction is known. See the 2026-08-01 note in
``docs/RESEARCH_ROADMAP.md`` Phase 1 item 1 for the full rationale and the
memory-leak fix that made even this reduced matrix tractable.

Reuses ``experiments.reproduce.runner`` unmodified via subprocess, exactly
like the sibling sweep scripts: resumable (skips combinations whose result
JSON already reports that client count's scaled round total as completed),
continues past a failing combination rather than aborting the whole
multi-hour sweep, and supports ``--force`` to ignore existing results and
rerun everything.

Client counts above the base scale to substantially more rounds (48 clients
-> 12x the base rounds), so this sweep costs much more wall-clock time than
``sweep_48_clients.py``'s fixed-round version -- expect a multi-hour run at
n=48 even with the reduced matrix and higher default parallelism.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTITION_MODES = ("homogeneous", "non-iid")
PRIVACY_MODES_SWEPT = ("global-dp", "metric-privacy")
AGGREGATION_METHODS_SWEPT = ("fedavg",)
CLIENT_COUNTS = (4, 8, 48)
BASE_NUM_CLIENTS = 4
BASE_ROUNDS = 20  # paper default (pyproject.toml num-server-rounds)
NOISE_MULTIPLIER = 0.05  # chosen from sweep_noise_multiplier.py's 8-client results
MAX_PARALLEL_CLIENTS = 8  # raised from 2 now that the MPS cache leak is fixed
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


def cleanup_orphaned_shm_managers() -> int:
    """Kill orphaned ``torch_shm_manager`` daemons left by past hard-killed runs.

    An OOM-kill, ``kill -9``, or crashed combo doesn't give PyTorch's
    shared-memory manager a chance to shut down cleanly -- it gets reparented
    to init (ppid 1) and lingers forever holding memory. One bad combo can
    leave dozens of these behind, and across a long multi-hour/multi-day
    sweep with several crashes they compound until the machine runs out of
    RAM/swap even with nothing currently training (observed: ~1,000 of these
    alone exhausted a 24GB machine's memory and swap). Filtering on ppid 1
    means this can never touch a live run's own workers, which always have a
    real parent, so it's safe to call at any point, including mid-sweep.
    """
    try:
        output = subprocess.run(
            ["ps", "-eo", "pid,ppid,comm"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        _log(f"WARN  shm cleanup: could not list processes ({error})")
        return 0

    orphaned_pids = []
    for line in output.splitlines()[1:]:
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid_text, ppid_text, comm = parts
        if ppid_text == "1" and comm.strip() == "torch_shm_manager":
            orphaned_pids.append(int(pid_text))

    for pid in orphaned_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    if orphaned_pids:
        _log(f"CLEANUP killed {len(orphaned_pids)} orphaned torch_shm_manager process(es)")
    return len(orphaned_pids)


def run_name(partition: str, privacy: str, aggregation: str, num_clients: int) -> str:
    return f"scalectrl__{partition}__{privacy}__{aggregation}__n{num_clients}"


def result_path(partition: str, privacy: str, aggregation: str, num_clients: int) -> Path:
    return OUTPUT_DIR / f"{run_name(partition, privacy, aggregation, num_clients)}.json"


def is_complete(path: Path, *, expected_rounds: int) -> bool:
    """Return whether ``path`` holds a valid, fully-completed result.

    Treats a missing, unparseable, or short-of-rounds file as incomplete, so
    a prior run that was killed mid-write (or mid-sweep) is rerun rather than
    silently accepted. Also requires the post-hoc evaluation artifact
    (``<run>.evaluation.json``, written by detailed_evaluation.py after
    training finishes) to exist -- training can complete all its rounds and
    still leave that step unrun/failed (observed: a device-mismatch bug in
    evaluate_state_dict's consistency check crashed after a full 40-round
    training run, leaving the round-complete run JSON behind with no
    evaluation file), which this function would otherwise treat as complete
    forever and silently never retry.
    """
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    history = data.get("server_evaluate_metrics", {})
    completed_rounds = [int(round_number) for round_number in history if int(round_number) > 0]
    if len(completed_rounds) < expected_rounds:
        return False
    evaluation_path = path.parent / f"{path.stem}.evaluation.json"
    return evaluation_path.exists()


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
    cleanup_orphaned_shm_managers()
    total = (
        len(CLIENT_COUNTS)
        * len(PARTITION_MODES)
        * len(PRIVACY_MODES_SWEPT)
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
            for privacy in PRIVACY_MODES_SWEPT:
                for aggregation in AGGREGATION_METHODS_SWEPT:
                    ok = run_one_combo(
                        partition,
                        privacy,
                        aggregation,
                        num_clients,
                        force=args.force,
                        max_parallel_clients=args.max_parallel_clients,
                    )
                    cleanup_orphaned_shm_managers()
                    completed += 1
                    if not ok:
                        failed.append(run_name(partition, privacy, aggregation, num_clients))
                    _log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")

    _log(f"Sweep finished: {completed}/{total} attempted, {len(failed)} failed")
    if failed:
        _log("Failed combinations: " + ", ".join(failed))


if __name__ == "__main__":
    main()
