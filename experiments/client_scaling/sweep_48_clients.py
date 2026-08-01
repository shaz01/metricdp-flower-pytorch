"""Sweep the privacy x aggregation matrix at 48 clients, using the noise
multiplier chosen from the 8-client noise sweep.

At the paper-default noise_multiplier=0.01, metric-privacy and global-dp
were near-indistinguishable (see ``sweep_noise_multiplier.py``): the
metric-privacy calibration factor (noise_multiplier / max pairwise
client-model distance) was a near-no-op because distance sat right around
1.0. noise_multiplier=0.05 was the sweet spot found there -- training
perturbation at that level pushes client-model distance to ~1.3-1.4,
which lowers metric-privacy's effective noise below global-dp's fixed
noise and produced a real accuracy advantage (+6.9pp homogeneous, +12.2pp
non-iid) at 8 clients, without collapsing training. This sweep checks
whether that advantage holds (or grows/shrinks) at 48 clients.

Scoped to fedavg and fedyogi (matching the reduced aggregation list used
in the 8-client scaling sweep; fedmedian/fedprox/fedopt dropped by request
there, and fedavgm further deferred to keep active sweeps small -- it
returns for the full 6-method paper run). Reuses
``experiments.reproduce.runner`` unmodified via
subprocess, exactly like ``sweep_8_clients.py`` and
``sweep_noise_multiplier.py``: resumable (skips combinations whose result
JSON already reports the paper-default number of completed rounds),
continues past a failing combination rather than aborting the whole
multi-hour sweep, and supports ``--force`` to ignore existing results and
rerun everything.

48 clients means substantially more per-round work than 8 clients, so
``--max-parallel-clients`` defaults higher here (4 vs. the runner's own
default of 2) to use more of this machine's 12 cores -- tune
``MAX_PARALLEL_CLIENTS`` down if memory pressure becomes an issue (48
clients x more concurrent Ray actors costs more RAM than 8 did).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.client_scaling.sweep_runner import run_sweep
from experiments.reproduce.matrix import Hyperparams, Matrix
from metricdp_pytorch.strategy_factory import PRIVACY_MODES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUM_CLIENTS = 48
MAX_PARALLEL_CLIENTS = 4
OUTPUT_DIR = PROJECT_ROOT / "results" / "48client_scaling"
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
    parser.add_argument(
        "--max-parallel-clients",
        type=int,
        default=MAX_PARALLEL_CLIENTS,
        help="cap simultaneous Ray actors to control memory use",
    )
    return parser



def main() -> None:
    args = _parser().parse_args()
    combos = MATRIX.list_combos(name_prefix="scaling48", num_clients=NUM_CLIENTS)
    run_sweep(
        combos,
        output_dir=OUTPUT_DIR,
        log_path=LOG_PATH,
        max_parallel_clients=args.max_parallel_clients,
        force=args.force,
        start_message=(
            f"Sweep starting: {len(combos)} combinations, num_clients={NUM_CLIENTS}, "
            f"noise_multiplier={MATRIX.noise_multipliers}, "
            f"max_parallel_clients={args.max_parallel_clients}, "
            f"force={args.force}"
        ),
    )


if __name__ == "__main__":
    main()
