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
from pathlib import Path

from experiments.client_scaling.sweep_runner import run_sweep
from experiments.reproduce.matrix import Hyperparams, Matrix
from metricdp_pytorch.strategy_factory import PRIVACY_MODES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUM_CLIENTS = 8
MAX_PARALLEL_CLIENTS = 2
OUTPUT_DIR = PROJECT_ROOT / "results" / "8client_scaling"
LOG_PATH = OUTPUT_DIR / "sweep_progress.log"

MATRIX = Matrix(
    partitions=("homogeneous", "non-iid"),
    privacy_modes=tuple(PRIVACY_MODES),
    aggregations=("fedavg", "fedyogi"),
    seeds=(42,),
    noise_multipliers=(0.05,),  # chosen from sweep_noise_multiplier.py's 8-client results
    hyperparams=Hyperparams(
        clipping_norm=5.0,
        rounds=20,
        local_epochs=5,
        batch_size=32,
        learning_rate=0.001,
        initialization_epochs=20,
    ),
    data_module="experiments.reproduce.dataset.alzheimer:create_data_module",
    model_module="experiments.reproduce.paper_cnn:create_model",
)

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
    combos = MATRIX.list_combos(name_prefix="scaling8", num_clients=NUM_CLIENTS)
    run_sweep(
        combos,
        output_dir=OUTPUT_DIR,
        log_path=LOG_PATH,
        max_parallel_clients=MAX_PARALLEL_CLIENTS,
        force=args.force,
        start_message=(
            f"Sweep starting: {len(combos)} combinations, "
            f"num_clients={NUM_CLIENTS}, force={args.force}"
        ),
    )


if __name__ == "__main__":
    main()
