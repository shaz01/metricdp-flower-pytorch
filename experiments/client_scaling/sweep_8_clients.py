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
import time
from pathlib import Path

from experiments.reproduce.matrix import Combo, Hyperparams
from experiments.reproduce.matrix.run_combo import run_one_combo
from metricdp_pytorch.strategy_factory import PRIVACY_MODES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTITION_MODES = ("homogeneous", "non-iid")
# fedmedian, fedprox, and fedopt were dropped from this sweep by request.
# fedavgm is further deferred (not permanently dropped) to keep the active
# diagnosis/redesign sweeps small; it returns for the full 6-method paper run.
SWEEP_AGGREGATION_METHODS = ("fedavg", "fedyogi")
NUM_CLIENTS = 8
SEED = 42
NOISE_MULTIPLIER = 0.01
CLIPPING_NORM = 5.0
ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
INITIALIZATION_EPOCHS = 20
MAX_PARALLEL_CLIENTS = 2
OUTPUT_DIR = PROJECT_ROOT / "results" / "8client_scaling"
LOG_PATH = OUTPUT_DIR / "sweep_progress.log"


def _log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


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
                combo = Combo(
                    name_prefix="scaling8",
                    num_clients=NUM_CLIENTS,
                    partition=partition,
                    privacy=privacy,
                    aggregation=aggregation,
                    seed=SEED,
                    hyperparams=Hyperparams(
                        noise_multiplier=NOISE_MULTIPLIER,
                        clipping_norm=CLIPPING_NORM,
                        rounds=ROUNDS,
                        local_epochs=LOCAL_EPOCHS,
                        batch_size=BATCH_SIZE,
                        learning_rate=LEARNING_RATE,
                        initialization_epochs=INITIALIZATION_EPOCHS,
                    ),
                )
                ok = run_one_combo(
                    combo,
                    output_dir=OUTPUT_DIR,
                    max_parallel_clients=MAX_PARALLEL_CLIENTS,
                    force=args.force,
                    log=_log,
                )
                completed += 1
                if not ok:
                    failed.append(combo.run_name())
                _log(f"PROGRESS {completed}/{total} ({len(failed)} failed so far)")

    _log(f"Sweep finished: {completed}/{total} attempted, {len(failed)} failed")
    if failed:
        _log("Failed combinations: " + ", ".join(failed))


if __name__ == "__main__":
    main()
