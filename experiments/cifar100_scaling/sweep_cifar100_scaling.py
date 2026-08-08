"""Train every setting at a fixed 100 clients and 250 rounds on full
100-class CIFAR-100.

Standalone client-count/privacy/partition study -- not gated on
docs/RESEARCH_ROADMAP.md's Phase 1-5 sequencing. Scoped to a single client
count (100, per project-owner direction) and a single round budget (250,
per project-owner direction -- shorter round-count grids tried earlier
were judged too short, and this replaced them with one longer run per
setting instead of a round-budget comparison): 1 * 1 * 3 * 2 * 1 = 6
combinations (privacy x partition x fedavg).

Model history: this sweep went through three model architectures before
settling on the current one. v1-v3 was a plain 3-block CNN (v2 briefly
added a 4th conv block, reverted after its natural per-round update
magnitude, ~29-30, exceeded clipping_norm=5.0 and froze every clipping
privacy mode). It was then replaced with a DenseNet+SELU architecture
(concatenative skip connections, GroupNorm(8), SELU with LeCun-normal init
and AlphaDropout, 553,220 params), built and verified specifically for
robustness to that kind of weight-space update-magnitude sensitivity --
including a second, independent freeze mode found in that generation:
*vanilla* (no DP mechanism at all) froze at n=128/homogeneous, because 128
highly-correlated client updates reinforced rather than averaged out (no
1/sqrt(n) reduction), producing a combined step too large for the model to
absorb on round 1. Per project-owner direction, that DenseNet+SELU
architecture was itself replaced with the current model: an adaptation of
the project supervisor's own CNNCIFAR100 reference architecture (3 blocks
of 2x[Conv3x3-GroupNorm-ReLU], channels 128/256/512, global-average-pooled
classifier, 4,631,268 params -- see experiments/reproduce/cifar100_cnn.py's
docstring for the adaptation record), now this project's one CIFAR-100
model. results/cifar100_scaling/ was cleared of all prior results on
2026-08-08 as part of this consolidation -- every earlier result (v1-v3's
plain-CNN runs, the DenseNet+SELU architecture's runs, and the supervisor
model's own earlier narrower client-count/round-count grid, formerly under
results/cifar100_scaling_supervisor/) described a model, grid, or
directory layout this sweep no longer uses, and none were kept. Starts
empty; --force is not required for this launch.

Caveat carried over unresolved from the shorter-round design:
NOISE_MULTIPLIER below was calibrated from a signal-update-norm measured
early in training (round 1 of a 3-round run). Client update magnitude
typically shrinks as training converges, which would drift the realized
noise-to-signal ratio upward over a 250-round run (constant noise against
a shrinking signal). Not re-measured at round 250 before this launch (that
would require the very run this sweep performs); worth revisiting if
250-round global-dp/metric-privacy results look worse, relative to
vanilla, than the r20 results already on record.

Reuses experiments.reproduce.runner unmodified via subprocess, exactly like
the sibling sweep scripts: resumable (skips combinations whose result JSON
already reports the target round count as completed), continues past a
failing combination rather than aborting the whole sweep, and supports
--force to ignore existing results and rerun everything.
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
CLIENT_COUNTS = (100,)
ROUND_COUNTS = (250,)
NOISE_MULTIPLIER = 0.0182  # calibrated for this model's 4,631,268 params,
# at n=100 (the only client count this sweep runs -- calibrating at the
# client count it actually uses is more correct than calibrating at an
# arbitrary different point).
# Derivation: target noise-to-signal ratio ~1, using noise_l2_norm ~= stdv *
# sqrt(param_count) and stdv = noise_multiplier * clipping_norm /
# num_sampled_clients. Measured directly on this model (n=100, homogeneous,
# global-dp, 3 rounds): signal update norm (post-clip, post-aggregation,
# pre-noise) = 1.956, param_count = 4,631,268, giving
# target_noise_multiplier = signal_norm * num_sampled_clients /
# (clipping_norm * sqrt(param_count)) = 1.956 * 100 / (5.0 * sqrt(4631268))
# ~= 0.0182. Confirmed via a follow-up run at this value: measured
# noise-to-signal ratio 1.001.
#
# Verification note: at this calibrated noise level, a 20-round real run
# (n=100, homogeneous, global-dp) was checked before launch for the
# clipping/magnitude freeze pattern found twice in this sweep's earlier
# DenseNet+SELU generation (see module docstring). Loss dropped for the
# first 2 rounds (4.72 -> 4.61), held close to ln(100)=4.605 for rounds 3-6
# (a slow warm-up, not the earlier freezes' pattern of staying flat for the
# entire run with no recovery), then broke free from round 7 onward and
# improved steadily through round 20 (loss 4.7154 -> 4.3958, accuracy
# 1.10% -> 2.74%). This model is simply slower to warm up than its
# predecessor (4.6M params, no skip connections, heavy 50% dropout, 100-way
# thin client splits) -- not frozen.
MAX_PARALLEL_CLIENTS = 16
OUTPUT_DIR = PROJECT_ROOT / "results" / "cifar100_scaling"
LOG_PATH = OUTPUT_DIR / "sweep_progress.log"

# Fixed hyperparameters across the whole sweep. Only --round-counts varies
# (num_clients is fixed at 100).
SEED = 42
CLIPPING_NORM = 5.0
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
INITIALIZATION_EPOCHS = 20
WEIGHT_DECAY = 5e-4
# Fixed LR, not cosine: a decaying LR would drop client updates below
# clipping_norm late in each run, drifting the noise-to-signal ratio back
# up and fighting the point of calibrating NOISE_MULTIPLIER above.
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
    convention of duplicating this helper per sweep script.
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

    Same contract as the sibling sweep scripts' is_complete: a missing,
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
    round_counts = tuple(args.round_counts)
    aggregation_methods = tuple(args.aggregation_methods)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_orphaned_shm_managers()
    total = (
        len(CLIENT_COUNTS)
        * len(round_counts)
        * len(PARTITION_MODES)
        * len(PRIVACY_MODES_SWEPT)
        * len(aggregation_methods)
    )
    _log(
        f"Sweep starting: {total} combinations, client_counts={CLIENT_COUNTS}, "
        f"round_counts={round_counts}, aggregation_methods={aggregation_methods}, "
        f"noise_multiplier={NOISE_MULTIPLIER}, "
        f"max_parallel_clients={args.max_parallel_clients}, force={args.force}"
    )

    completed = 0
    failed: list[str] = []
    for num_clients in CLIENT_COUNTS:
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
