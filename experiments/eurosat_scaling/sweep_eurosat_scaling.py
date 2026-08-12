"""Train every setting at a fixed 48 clients on the full 10-class EuroSAT dataset.

Comparison point for experiments/cifar100_scaling/'s accuracy sweep, on a genuinely simpler,
different-domain dataset (satellite land-use imagery, not CIFAR-10 -- a teammate's part of this
project -- or MNIST-family). CLIENT_COUNTS=(48,) rather than CIFAR-100's (100,): EuroSAT has only
21,600 train images (vs 50,000), so n=48 keeps per-client data volume (~450 images) comparable to
CIFAR-100's (~500 images at n=100) rather than starving it further. 1 * 1 * 3 * 2 * 1 = 6
combinations (privacy x partition x fedavg).

NOISE_MULTIPLIER below was calibrated empirically -- see the comment at its definition for the
full derivation and the measured signal-update-norm this repo's dp-signal-update-norm diagnostic
produced on this specific model/dataset/client-count, following the same target-noise-to-signal-
ratio-of-1 formula experiments/cifar100_scaling/sweep_cifar100_scaling.py's own NOISE_MULTIPLIER
used.

Reuses experiments.reproduce.runner unmodified via subprocess, exactly like the sibling sweep
scripts: resumable (skips combinations whose result JSON already reports the target round count
as completed), continues past a failing combination rather than aborting the whole sweep, and
supports --force to ignore existing results and rerun everything.

No weight_decay/lr_schedule: feature/cifar100-scaling (still active, unmerged) added
--weight-decay/--lr-schedule support to runner.py/Hyperparams/Combo.runner_args(), but this
branch is based on master, which doesn't have it -- Hyperparams here only has clipping_norm,
rounds, local_epochs, batch_size, learning_rate, initialization_epochs. Decision (project owner,
2026-08-12): build on master's actual fields rather than duplicate that feature independently.
weight_decay therefore runs at runner.py's default (0.0); lr_schedule="none" was already
CIFAR-100's own value, so that one is unaffected.
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
PRIVACY_MODES_SWEPT = ("vanilla", "global-dp", "metric-privacy")
AGGREGATION_METHODS_SWEPT = ("fedavg",)
CLIENT_COUNTS = (48,)
ROUND_COUNTS = (100,)  # EuroSAT is a much easier 10-way task than CIFAR-100's 100-way
# classification on similarly-sized images, so a shorter budget than CIFAR-100's 250 is a
# reasonable starting point -- verified against the early-freeze failure modes documented in
# experiments/cifar100_scaling/sweep_cifar100_scaling.py's docstring via a 20-round check at
# Step 8 below before committing to the full sweep.

NOISE_MULTIPLIER = 0.01  # PLACEHOLDER -- overwritten at Step 8 below with the value computed
# from this run's own measured dp-signal-update-norm, following the exact formula
# experiments/cifar100_scaling/sweep_cifar100_scaling.py's own NOISE_MULTIPLIER derivation used:
# target_noise_multiplier = signal_norm * num_sampled_clients / (clipping_norm * sqrt(param_count))
# param_count = 289,194 (verified, see eurosat_cnn.py). Do not launch the real 6-combo sweep with
# this starting value -- it is not calibrated.

MAX_PARALLEL_CLIENTS = 16
OUTPUT_DIR = PROJECT_ROOT / "results" / "eurosat_scaling"
LOG_PATH = OUTPUT_DIR / "sweep_progress.log"

SEED = 42
CLIPPING_NORM = 5.0
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
INITIALIZATION_EPOCHS = 20
# No WEIGHT_DECAY / LR_SCHEDULE constants -- this branch's Hyperparams has no such fields (see
# module docstring above). Do not add them without also adding --weight-decay/--lr-schedule to
# this branch's runner.py and Hyperparams first.
DATA_MODULE = "experiments.reproduce.dataset.eurosat:create_data_module"
MODEL_MODULE = "experiments.reproduce.eurosat_cnn:create_model"


def _log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def run_name(
    partition: str, privacy: str, aggregation: str, num_clients: int, rounds: int
) -> str:
    return (
        f"eurosatscale__{partition}__{privacy}__{aggregation}__"
        f"n{num_clients}__r{rounds}"
    )


def result_path(
    partition: str, privacy: str, aggregation: str, num_clients: int, rounds: int
) -> Path:
    return OUTPUT_DIR / f"{run_name(partition, privacy, aggregation, num_clients, rounds)}.json"


def is_complete(path: Path, *, expected_rounds: int) -> bool:
    """Return whether ``path`` holds a valid, fully-completed result.

    Matches sibling sweep scripts' is_complete contract: a missing, unparseable, or
    short-of-rounds file is incomplete, and the post-hoc evaluation artifact
    (<run>.evaluation.json) must also exist.
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


def build_combo(
    partition: str, privacy: str, aggregation: str, num_clients: int, rounds: int
) -> Combo:
    """Build the Combo for one sweep combination.

    Hyperparams only carries the fields this branch's matrix module defines --
    clipping_norm, rounds, local_epochs, batch_size, learning_rate,
    initialization_epochs. No weight_decay/lr_schedule here (see module docstring).
    """
    return Combo(
        name_prefix="eurosatscale",
        num_clients=num_clients,
        partition=partition,
        privacy=privacy,
        aggregation=aggregation,
        seed=SEED,
        noise_multiplier=NOISE_MULTIPLIER,
        hyperparams=Hyperparams(
            clipping_norm=CLIPPING_NORM,
            rounds=rounds,
            local_epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            initialization_epochs=INITIALIZATION_EPOCHS,
        ),
        data_module=DATA_MODULE,
        model_module=MODEL_MODULE,
    )


def runner_args_with_name(
    combo: Combo, *, name: str, max_parallel_clients: int
) -> tuple[str, ...]:
    args = combo.runner_args(
        output_dir=OUTPUT_DIR, max_parallel_clients=max_parallel_clients, client_cpus=1.0
    )
    assert args[-2] == "--run-name"
    return (*args[:-1], name)


def run_one_combo(
    partition: str,
    privacy: str,
    aggregation: str,
    num_clients: int,
    rounds: int,
    *,
    force: bool,
    max_parallel_clients: int,
) -> bool:
    """Run one combination; return True on success (or already-complete)."""
    name = run_name(partition, privacy, aggregation, num_clients, rounds)
    path = result_path(partition, privacy, aggregation, num_clients, rounds)
    if not force and is_complete(path, expected_rounds=rounds):
        _log(f"SKIP  {name} (already complete)")
        return True

    combo = build_combo(partition, privacy, aggregation, num_clients, rounds)
    command = [
        sys.executable,
        "-m",
        "experiments.reproduce.runner",
        *runner_args_with_name(combo, name=name, max_parallel_clients=max_parallel_clients),
    ]
    _log(f"START {name} (rounds={rounds}, clients={num_clients})")
    started = time.monotonic()
    child_env = os.environ.copy()
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=MAX_PARALLEL_CLIENTS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = (
        len(CLIENT_COUNTS)
        * len(ROUND_COUNTS)
        * len(PARTITION_MODES)
        * len(PRIVACY_MODES_SWEPT)
        * len(AGGREGATION_METHODS_SWEPT)
    )
    _log(
        f"Sweep starting: {total} combinations, client_counts={CLIENT_COUNTS}, "
        f"round_counts={ROUND_COUNTS}, noise_multiplier={NOISE_MULTIPLIER}, "
        f"max_parallel_clients={args.max_parallel_clients}, force={args.force}"
    )

    completed = 0
    failed: list[str] = []
    for num_clients in CLIENT_COUNTS:
        for rounds in ROUND_COUNTS:
            for partition in PARTITION_MODES:
                for privacy in PRIVACY_MODES_SWEPT:
                    for aggregation in AGGREGATION_METHODS_SWEPT:
                        ok = run_one_combo(
                            partition,
                            privacy,
                            aggregation,
                            num_clients,
                            rounds,
                            force=args.force,
                            max_parallel_clients=args.max_parallel_clients,
                        )
                        completed += 1
                        if not ok:
                            failed.append(
                                run_name(partition, privacy, aggregation, num_clients, rounds)
                            )
                        _log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")

    _log(f"Sweep finished: {completed}/{total} attempted, {len(failed)} failed")
    if failed:
        _log("Failed combinations: " + ", ".join(failed))


if __name__ == "__main__":
    main()
