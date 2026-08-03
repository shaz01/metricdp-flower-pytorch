"""
Contests paper Tables 1-3. Is not for contesting paper's Table 9.
"""
from pathlib import Path

from experiments.cia.attack_runner import run_attack
from experiments.cia.shadow_dataset import clean_shadow_dataset, noisy_shadow_dataset
from experiments.reproduce.matrix import Matrix, Hyperparams, Combo
from metricdp_pytorch.utils.device import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "results" / "reproduce"

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

CLEAN_SHADOW_DATASET = lambda combo: clean_shadow_dataset(
    combo,
    target_partition_id=TARGET_PARTITION_ID,
    shadow_fraction=SHADOW_FRACTION,
)

# TODO - use after multi-round cia is implemented
NOISY_SHADOW_DATASET = lambda combo: noisy_shadow_dataset(
    combo,
    target_partition_id=TARGET_PARTITION_ID,
    shadow_fraction=SHADOW_FRACTION,
    std_fraction=NOISE_STD_FRACTION,
)


def main() -> None:
    combos = MATRIX.list_combos(name_prefix="contest", num_clients=NUM_CLIENTS)
    results = run_attack(
        combos=combos,
        output_dir=OUTPUT_DIR,
        log_path=OUTPUT_DIR / 'progress.log',
        max_parallel_clients=4,
        force=False,
        start_message=f"Reproduction starting: {len(combos)} combinations",
        data_module_factory=CLEAN_SHADOW_DATASET,
        device=resolve_device(),
        checkpoint_rounds=CHECKPOINT_ROUNDS,
        report_name="contest.json",
    )
    for result in results:
        print(
            f"round={result.server_round:2d} {result.partition_mode:12s} "
            f"{result.privacy:15s} {result.aggregation:8s} "
            f"agg={result.aggregated_test_loss:.3f} "
            f"target={result.target_shadow_loss:.3f} "
            f"shadow_n={result.shadow_size} diff={result.difference_pct:.3f}%"
        )
