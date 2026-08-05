"""Sweep the privacy x client-count matrix under a rounds-fixed, epoch-scaled
constant-compute control -- the successor to ``sweep_scale_controlled.py``
(v1), whose ``rounds(n) = 5n`` design turned out to have its own confound.

v1's report (``reports/constant_compute_scaling.md``) found accuracy at 48
clients dropping far below the fixed-20-round baseline, and initially
attributed this to accumulated DP noise growing with round count. That
explanation didn't hold up: `rounds(n) * sigma(n)` is an exact constant for
global-dp under v1's formula (Flower's ``compute_stdv`` already scales sigma
by ``1 / num_clients``, which exactly offsets the round-count increase), and
metric-privacy's actual logged noise trajectory was *not* monotonically worse
at higher n either. What did line up: v1 held ``--local-epochs``/
``--batch-size`` fixed while scaling ``--rounds`` with ``num_clients``, so at
48 clients (86-sample homogeneous shards, batch_size=32) each local epoch was
only 2-4 mini-batches -- v1's total gradient-step count across the whole run
was roughly preserved, but delivered as 240 rounds of tiny batches (240
aggregation/noise-injection interruptions) instead of 20 rounds of
substantial batches (20 interruptions). Total-step-count parity does not
imply comparable training dynamics when the step count is fragmented across
12x more, much shorter local phases.

This version targets aggregation-frequency parity directly instead of only
total-step-count parity: hold ``--rounds`` fixed at ``BASE_ROUNDS`` for every
client count, and scale ``--local-epochs`` proportionally instead:

    local_epochs(n) = round(BASE_LOCAL_EPOCHS * n / BASE_NUM_CLIENTS)

Every client count now completes the same 20 rounds (same number of
aggregation/DP-noise-injection events across the whole run), while still
approximately preserving total local gradient steps per client
(``rounds * local_epochs(n) * batches_per_epoch(n)``) via more local epochs
per round instead of more rounds. Held fixed: ``--batch-size`` (32),
``noise_multiplier`` (0.05, unchanged from v1), matrix scope (``global-dp``
+ ``metric-privacy``, ``fedavg`` only, ``num_clients in {4, 8, 48}``) --
identical to v1's scope, so the two reports are otherwise apples-to-apples.

Trades one distortion for a different, more standard one: at n=48,
``local_epochs(48) = 60`` means each round trains 60 epochs over as few as
34-137 local samples before aggregating -- a lot of repeated passes over a
tiny, possibly non-representative local shard, i.e. classic FL client-drift
risk from over-training small non-iid shards. That's a well-understood
FL phenomenon in its own right, not a hidden confound in the experimental
design the way v1's interruption-frequency mismatch was -- but it's still
worth watching for in the results.

Shares its resumable/orphan-cleanup/checkpoint-resume infrastructure's
*pattern* with ``sweep_scale_controlled.py`` but doesn't import from it, to
keep this file self-contained and to avoid the two scripts' output
directories/state ever being able to cross-contaminate.
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

from experiments.reproduce.matrix import Combo, Hyperparams
from metricdp_pytorch.strategy_factory import AGGREGATION_METHODS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTITION_MODES = ("homogeneous", "non-iid")
PRIVACY_MODES_SWEPT = ("global-dp", "metric-privacy")
AGGREGATION_METHODS_SWEPT = ("fedavg",)
CLIENT_COUNTS = (4, 8, 48)
BASE_NUM_CLIENTS = 4
BASE_ROUNDS = 20  # held fixed for every client count -- see module docstring
BASE_LOCAL_EPOCHS = 5  # paper default; scaled up with client count below
NOISE_MULTIPLIER = 0.05  # unchanged from sweep_scale_controlled.py (v1)
MAX_PARALLEL_CLIENTS = 8
OUTPUT_DIR = PROJECT_ROOT / "results" / "scale_controlled_epochs"
LOG_PATH = OUTPUT_DIR / "sweep_progress.log"

# Paper-default hyperparameters held fixed across the whole sweep (only
# --local-epochs varies, via local_epochs_for()) -- same values as
# sweep_8_clients.py / pyproject.toml's [tool.flwr.app.config] defaults.
SEED = 42
CLIPPING_NORM = 5.0
BATCH_SIZE = 32
LEARNING_RATE = 0.001
INITIALIZATION_EPOCHS = 20
DATA_MODULE = "experiments.reproduce.dataset.alzheimer:create_data_module"
MODEL_MODULE = "experiments.reproduce.paper_cnn:create_model"


def rounds_for(num_clients: int) -> int:
    """Rounds are fixed for every client count -- this is the whole point."""
    del num_clients
    return BASE_ROUNDS


def local_epochs_for(num_clients: int) -> int:
    """Scale local epochs so per-client total gradient steps stay ~constant.

    Same proportional-approximation fidelity as v1's round-scaling formula
    (average shard size, not per-client exact accounting), just applied to
    epochs instead of rounds so aggregation frequency stays constant too.
    """
    return round(BASE_LOCAL_EPOCHS * num_clients / BASE_NUM_CLIENTS)


def _log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def cleanup_orphaned_shm_managers() -> int:
    """Kill orphaned daemons left behind by past hard-killed training runs.

    See ``sweep_scale_controlled.py``'s copy of this function for the full
    rationale; duplicated here (not imported) to keep this script
    self-contained per the module docstring.
    """
    try:
        output = subprocess.run(
            ["ps", "-eo", "pid,ppid,args"],
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
        pid_text, ppid_text, args = parts
        if ppid_text != "1":
            continue
        if args == "torch_shm_manager" or "paper-reproduction-" in args:
            orphaned_pids.append(int(pid_text))

    for pid in orphaned_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    if orphaned_pids:
        _log(f"CLEANUP killed {len(orphaned_pids)} orphaned training process(es)")
    return len(orphaned_pids)


def run_name(partition: str, privacy: str, aggregation: str, num_clients: int) -> str:
    return f"scalectrlep__{partition}__{privacy}__{aggregation}__n{num_clients}"


def result_path(partition: str, privacy: str, aggregation: str, num_clients: int) -> Path:
    return OUTPUT_DIR / f"{run_name(partition, privacy, aggregation, num_clients)}.json"


def is_complete(path: Path, *, expected_rounds: int) -> bool:
    """Return whether ``path`` holds a valid, fully-completed result.

    Treats a missing, unparseable, or short-of-rounds file as incomplete, and
    also requires the post-hoc evaluation artifact
    (``<run>.evaluation.json``) to exist -- see
    ``sweep_scale_controlled.py``'s copy for the incident that motivated
    this second check.
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


def resumable_checkpoint(path: Path, *, expected_rounds: int) -> Path | None:
    """Return the checkpoint path if only the evaluation step needs (re)running.

    See ``sweep_scale_controlled.py``'s copy for the full rationale (training
    completing but the evaluation step failing/never running, with
    ``server.py`` always saving a checkpoint before evaluation to make this
    possible).
    """
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    history = data.get("server_evaluate_metrics", {})
    completed_rounds = [int(round_number) for round_number in history if int(round_number) > 0]
    if len(completed_rounds) < expected_rounds:
        return None
    evaluation_path = path.parent / f"{path.stem}.evaluation.json"
    if evaluation_path.exists():
        return None
    checkpoint_path = path.parent / f"{path.stem}.pt"
    return checkpoint_path if checkpoint_path.exists() else None


def resume_evaluation_only(name: str, path: Path, checkpoint_path: Path) -> bool:
    """Re-run just the evaluation step from a saved checkpoint; return success."""
    evaluation_path = path.parent / f"{path.stem}.evaluation.json"
    predictions_path = path.parent / f"{path.stem}.predictions.npz"
    command = [
        sys.executable,
        "-m",
        "experiments.reproduce.detailed_evaluation",
        "--model",
        str(checkpoint_path),
        "--run-json",
        str(path),
        "--evaluation-json",
        str(evaluation_path),
        "--predictions",
        str(predictions_path),
        "--delete-model-on-success",
    ]
    _log(f"RESUME-EVAL {name} (training already complete, retrying evaluation only)")
    started = time.monotonic()
    result = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.monotonic() - started
    if result.returncode == 0:
        _log(f"DONE  {name} ({elapsed:.1f}s, evaluation-only resume)")
        return True
    _log(f"FAILED {name} (evaluation-only resume, exit={result.returncode}, {elapsed:.1f}s)")
    return False


def runner_args_with_name(
    combo: Combo, *, name: str, max_parallel_clients: int
) -> tuple[str, ...]:
    """Build this combo's runner.py CLI args under this sweep's own run name.

    See ``sweep_scale_controlled.py``'s copy for the full rationale
    (duplicated, not imported, per this module's self-containment
    docstring) -- delegates to ``Combo.runner_args()`` so every arg
    runner.py now requires (--local-epochs/--seed/--clipping-norm/
    --model-module, missing here before this fix -- see the 2026-08-05 note
    in STATUS.md) is included, then swaps Combo's trailing ``--run-name``
    pair for this sweep's own naming so ``result_path()``/``is_complete()``
    keep matching what actually lands on disk.
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
    *,
    force: bool,
    max_parallel_clients: int,
) -> bool:
    """Run one combination; return True on success (or already-complete)."""
    name = run_name(partition, privacy, aggregation, num_clients)
    path = result_path(partition, privacy, aggregation, num_clients)
    rounds = rounds_for(num_clients)
    local_epochs = local_epochs_for(num_clients)
    if not force and is_complete(path, expected_rounds=rounds):
        _log(f"SKIP  {name} (already complete)")
        return True

    if not force:
        checkpoint_path = resumable_checkpoint(path, expected_rounds=rounds)
        if checkpoint_path is not None:
            return resume_evaluation_only(name, path, checkpoint_path)

    combo = Combo(
        name_prefix="scalectrlep",
        num_clients=num_clients,
        partition=partition,
        privacy=privacy,
        aggregation=aggregation,
        seed=SEED,
        noise_multiplier=NOISE_MULTIPLIER,
        hyperparams=Hyperparams(
            clipping_norm=CLIPPING_NORM,
            rounds=rounds,
            local_epochs=local_epochs,
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
    _log(f"START {name} (rounds={rounds}, local_epochs={local_epochs})")
    started = time.monotonic()
    child_env = os.environ.copy()
    # Same determinism env vars experiments/reproduce/matrix/run_combo.py sets
    # for the sibling sweeps -- required for the reply-order/floating-point
    # determinism fix this whole redo exists to rely on (see STATUS.md).
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
    parser.add_argument(
        "--client-counts",
        type=int,
        nargs="+",
        default=list(CLIENT_COUNTS),
        help=(
            "subset of client counts to run (default: all of "
            f"{CLIENT_COUNTS}); lets one sweep be split across machines"
        ),
    )
    parser.add_argument(
        "--aggregation-methods",
        choices=AGGREGATION_METHODS,
        nargs="+",
        default=list(AGGREGATION_METHODS_SWEPT),
        help=(
            "subset of aggregation methods to run (default: "
            f"{AGGREGATION_METHODS_SWEPT}); lets e.g. fedyogi be added on a "
            "separate machine from the default fedavg sweep"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    client_counts = tuple(args.client_counts)
    aggregation_methods = tuple(args.aggregation_methods)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_orphaned_shm_managers()
    total = (
        len(client_counts)
        * len(PARTITION_MODES)
        * len(PRIVACY_MODES_SWEPT)
        * len(aggregation_methods)
    )
    _log(
        f"Sweep starting: {total} combinations, client_counts={client_counts}, "
        f"aggregation_methods={aggregation_methods}, "
        f"rounds={BASE_ROUNDS} (fixed), "
        f"local_epochs={[local_epochs_for(n) for n in client_counts]}, "
        f"noise_multiplier={NOISE_MULTIPLIER}, "
        f"max_parallel_clients={args.max_parallel_clients}, force={args.force}"
    )

    completed = 0
    failed: list[str] = []
    for num_clients in client_counts:
        for partition in PARTITION_MODES:
            for privacy in PRIVACY_MODES_SWEPT:
                for aggregation in aggregation_methods:
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
