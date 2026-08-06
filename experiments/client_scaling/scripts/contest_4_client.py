"""
For trying to reproduce or contesting paper Tables 1-3.
"""
# TODO: just do IN-remove CIA experiment?
from pathlib import Path

from experiments.client_scaling.sweep_runner import run_sweep
from experiments.reproduce.matrix import Matrix, Hyperparams

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "results" / "contest_4_clients"
LOG_PATH = OUTPUT_DIR / "progress.log"

NUM_CLIENTS = 4
TARGET_PARTITION_ID = 0
SHADOW_FRACTION = 0.10
NOISE_STD_FRACTION = 0.10

ROUNDS = 20
CHECKPOINT_ROUNDS = (1, ROUNDS)  # Single round CIA - first round and last round.

MATRIX = Matrix(
    partitions=("homogeneous",),
    privacy_modes=("vanilla", "global-dp", "metric-privacy"),
    aggregations=("fedavg",),
    seeds=(42, 43, 44, 45, 46),
    noise_multipliers=(0.01,),
    hyperparams=Hyperparams(
        clipping_norm=5.0,
        rounds=ROUNDS,
        local_epochs=5,
        batch_size=32,
        learning_rate=0.001,
        initialization_epochs=20,
    ),
    data_module="experiments.reproduce.dataset.alzheimer:create_data_module",
    model_module="experiments.reproduce.paper_cnn:create_model",
)

def main() -> None:
    combos = MATRIX.list_combos(name_prefix="contest", num_clients=NUM_CLIENTS)
    run_sweep(
        combos,
        output_dir=OUTPUT_DIR,
        log_path=LOG_PATH,
        max_parallel_clients=4,
        force=False,
        start_message=(
            f"Sweep starting: {len(combos)} combinations, "
            f"num_clients={NUM_CLIENTS}"
        ),
    )
