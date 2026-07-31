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
import time
from pathlib import Path

from experiments.reproduce.matrix import Combo, Hyperparams
from experiments.reproduce.matrix.run_combo import run_one_combo

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PARTITION_MODES = ("homogeneous", "non-iid")
PRIVACY_MODES_SWEPT = ("global-dp", "metric-privacy")
AGGREGATION_METHODS_SWEPT = ("fedavg",)
NOISE_MULTIPLIERS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
NUM_CLIENTS = 8
SEED = 42
CLIPPING_NORM = 5.0
ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
INITIALIZATION_EPOCHS = 20
MAX_PARALLEL_CLIENTS = 2
OUTPUT_DIR = PROJECT_ROOT / "results" / "noise_sweep"
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
                    combo = Combo(
                        name_prefix="noise8",
                        num_clients=NUM_CLIENTS,
                        partition=partition,
                        privacy=privacy,
                        aggregation=aggregation,
                        seed=SEED,
                        hyperparams=Hyperparams(
                            noise_multiplier=noise_multiplier,
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
