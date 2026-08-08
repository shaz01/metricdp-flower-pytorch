"""Sweep client count x round budget on full 100-class CIFAR-100.

Standalone client-count/round-budget scaling study -- not gated on
docs/RESEARCH_ROADMAP.md's Phase 1-5 sequencing. Grids num_clients in
{8, 64, 128, 256} against rounds in {20, 60, 120}, for each of
vanilla/global-dp/metric-privacy and each of homogeneous/non-iid
partitioning, on fedavg only: 4 * 3 * 3 * 2 * 1 = 72 combinations.

Unlike every other dataset plugin used by the sibling sweeps in
experiments/client_scaling/ (all reduced to a four-class subset), this uses
the full 100-class experiments.reproduce.dataset.cifar100 plugin -- a
genuinely harder task with a much larger model, so noise_multiplier is
recalibrated to 0.0025 rather than the paper's 0.01 or this sweep's own v1
value of 0.05; see the NOISE_MULTIPLIER constant below for the actual
derivation.

Reuses experiments.reproduce.runner unmodified via subprocess, exactly like
the sibling sweep scripts: resumable (skips combinations whose result JSON
already reports the target round count as completed), continues past a
failing combination rather than aborting the whole multi-day sweep, and
supports --force to ignore existing results and rerun everything.

results/cifar100_scaling/ was cleared on 2026-08-08 when the model was
replaced with a DenseNet+SELU architecture (see
docs/superpowers/specs/2026-08-08-cifar100-densenet-selu-design.md) --
the v1/v2/v3 3-block-CNN results it held are not comparable to this
architecture and were discarded rather than kept alongside it. Starts
empty; --force is not required for this launch, but remains available for
any future re-launch that needs to overwrite completed results.
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
PRIVACY_MODES_SWEPT = ("vanilla", "global-dp", "metric-privacy")
AGGREGATION_METHODS_SWEPT = ("fedavg",)
CLIENT_COUNTS = (8, 64, 128, 256)
ROUND_COUNTS = (20, 60, 120)
NOISE_MULTIPLIER = 0.0025  # recalibrated for the v2 model's ~2.76M params;
# see docs/superpowers/specs/2026-08-07-cifar100-v2-accuracy-design.md for
# the derivation (target noise-to-signal ratio ~1 at n=8, using
# noise_l2_norm ~= stdv * sqrt(param_count) and stdv = noise_multiplier *
# clipping_norm / num_sampled_clients). v1 used 0.05, which produced a
# ~22x noise-to-signal ratio and crushed global-dp to ~4% accuracy at n=8.
MAX_PARALLEL_CLIENTS = 16
OUTPUT_DIR = PROJECT_ROOT / "results" / "cifar100_scaling"
LOG_PATH = OUTPUT_DIR / "sweep_progress.log"

# Fixed hyperparameters across the whole sweep -- paper defaults except
# noise_multiplier (see module docstring). Only --rounds/--num-clients vary.
SEED = 42
CLIPPING_NORM = 5.0
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
INITIALIZATION_EPOCHS = 20
WEIGHT_DECAY = 5e-4
# Fixed LR, not cosine: DP noise is constant all run, but a decaying LR drops
# client updates below clipping_norm late in each run (~round 16/20, 45/60,
# 90/120), so clipping stops binding and the noise-to-signal ratio drifts back
# up toward ~19x in the final ~25% of the round budget -- fighting the whole
# point of retuning NOISE_MULTIPLIER above. See the review that flagged this.
LR_SCHEDULE = "none"
DATA_MODULE = "experiments.reproduce.dataset.cifar100:create_data_module"
MODEL_MODULE = "experiments.reproduce.cifar100_cnn:create_model"


def _log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def cleanup_orphaned_shm_managers() -> int:
    """Kill orphaned daemons left behind by past hard-killed training runs.

    Copied from sweep_scale_controlled.py (see that file's docstring for the
    full rationale) rather than shared -- matches this repo's existing
    convention of duplicating this helper per sweep script
    (sweep_scale_controlled_epochs.py does the same).
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


def run_name(
    partition: str, privacy: str, aggregation: str, num_clients: int, rounds: int
) -> str:
    return (
        f"cifar100scale__{partition}__{privacy}__{aggregation}__"
        f"n{num_clients}__r{rounds}"
    )


def result_path(
    partition: str, privacy: str, aggregation: str, num_clients: int, rounds: int
) -> Path:
    return OUTPUT_DIR / f"{run_name(partition, privacy, aggregation, num_clients, rounds)}.json"


def is_complete(path: Path, *, expected_rounds: int) -> bool:
    """Return whether ``path`` holds a valid, fully-completed result.

    Same contract as sweep_scale_controlled.py's is_complete: a missing,
    unparseable, or short-of-rounds file is incomplete, and the post-hoc
    evaluation artifact (<run>.evaluation.json) must also exist.
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
    """Return the checkpoint path if only the evaluation step needs (re)running."""
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
    """Build this combo's runner.py CLI args under this sweep's own run name."""
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

    if not force:
        checkpoint_path = resumable_checkpoint(path, expected_rounds=rounds)
        if checkpoint_path is not None:
            return resume_evaluation_only(name, path, checkpoint_path)

    combo = Combo(
        name_prefix="cifar100scale",
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
            weight_decay=WEIGHT_DECAY,
            lr_schedule=LR_SCHEDULE,
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
        "--round-counts",
        type=int,
        nargs="+",
        default=list(ROUND_COUNTS),
        help=f"subset of round counts to run (default: all of {ROUND_COUNTS})",
    )
    parser.add_argument(
        "--aggregation-methods",
        choices=AGGREGATION_METHODS,
        nargs="+",
        default=list(AGGREGATION_METHODS_SWEPT),
        help=(
            "subset of aggregation methods to run (default: "
            f"{AGGREGATION_METHODS_SWEPT})"
        ),
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    client_counts = tuple(args.client_counts)
    round_counts = tuple(args.round_counts)
    aggregation_methods = tuple(args.aggregation_methods)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_orphaned_shm_managers()
    total = (
        len(client_counts)
        * len(round_counts)
        * len(PARTITION_MODES)
        * len(PRIVACY_MODES_SWEPT)
        * len(aggregation_methods)
    )
    _log(
        f"Sweep starting: {total} combinations, client_counts={client_counts}, "
        f"round_counts={round_counts}, aggregation_methods={aggregation_methods}, "
        f"noise_multiplier={NOISE_MULTIPLIER}, "
        f"max_parallel_clients={args.max_parallel_clients}, force={args.force}"
    )

    completed = 0
    failed: list[str] = []
    for num_clients in client_counts:
        for rounds in round_counts:
            for partition in PARTITION_MODES:
                for privacy in PRIVACY_MODES_SWEPT:
                    for aggregation in aggregation_methods:
                        ok = run_one_combo(
                            partition,
                            privacy,
                            aggregation,
                            num_clients,
                            rounds,
                            force=args.force,
                            max_parallel_clients=args.max_parallel_clients,
                        )
                        cleanup_orphaned_shm_managers()
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
