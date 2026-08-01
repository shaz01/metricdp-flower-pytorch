"""Evaluate CIA at rounds 1 and 20 of the same 48-client trajectories.

Each privacy × aggregation × partition combination is trained once for 20
rounds using the concluded client-scaling hyperparameters. The attack is then
evaluated from checkpoints at rounds 1 and 20; round 1 is not retrained as a
separate model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.cia.attack_runner import run_attack
from experiments.cia.datasets.shadow import ShadowDataModule
from experiments.cia.result import CiaResult
from experiments.reproduce.dataset.alzheimer import AlzheimerDataModule
from experiments.reproduce.matrix import Combo, Hyperparams, Matrix
from metricdp_pytorch.strategy_factory import PRIVACY_MODES
from metricdp_pytorch.utils.device import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NUM_CLIENTS = 48
TARGET_PARTITION_ID = 0
SEED = 42
BATCH_SIZE = 32
CHECKPOINT_ROUNDS = (1, 20)
MAX_PARALLEL_CLIENTS = 4
OUTPUT_DIR = PROJECT_ROOT / "results" / "cia_client_scaling"

MATRIX = Matrix(
    partitions=("homogeneous", "non-iid"),
    privacy_modes=tuple(PRIVACY_MODES),
    aggregations=("fedavg", "fedyogi"),
    seeds=(SEED,),
    noise_multipliers=(0.05,),
    hyperparams=Hyperparams(
        clipping_norm=5.0,
        rounds=20,
        local_epochs=5,
        batch_size=BATCH_SIZE,
        learning_rate=0.001,
        initialization_epochs=20,
    ),
    data_module="experiments.cia.datasets.shadow:create_shadow_data_module",
    model_module="experiments.reproduce.paper_cnn:create_model",
)


def build_combos() -> list[Combo]:
    return MATRIX.list_combos(
        name_prefix="cia_scaling", num_clients=NUM_CLIENTS
    )


def _data_module_for(combo: Combo) -> ShadowDataModule:
    return ShadowDataModule(
        AlzheimerDataModule(),
        num_clients=combo.num_clients,
        target_partition_id=TARGET_PARTITION_ID,
        shadow_fraction=0.10,
        partition_mode=combo.partition,
        partition_profile="auto",
    )


def run_client_scaling_cia(
    *,
    output_dir: Path = OUTPUT_DIR,
    max_parallel_clients: int = MAX_PARALLEL_CLIENTS,
    force: bool = False,
) -> list[CiaResult]:
    combos = build_combos()
    return run_attack(
        combos,
        output_dir=output_dir,
        log_path=output_dir / "attack_progress.log",
        max_parallel_clients=max_parallel_clients,
        force=force,
        start_message=(
            f"48-client CIA starting: {len(combos)} trajectories, "
            f"checkpoint_rounds={CHECKPOINT_ROUNDS}, force={force}"
        ),
        data_module_factory=_data_module_for,
        device=resolve_device(),
        batch_size=BATCH_SIZE,
        seed=SEED,
        checkpoint_rounds=CHECKPOINT_ROUNDS,
        report_name="cia_client_scaling.json",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--max-parallel-clients", type=int, default=MAX_PARALLEL_CLIENTS
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    results = run_client_scaling_cia(
        output_dir=args.output_dir,
        max_parallel_clients=args.max_parallel_clients,
        force=args.force,
    )
    for result in results:
        print(
            f"round={result.server_round:2d} {result.partition_mode:12s} "
            f"{result.privacy:15s} {result.aggregation:8s} "
            f"agg={result.aggregated_test_loss:.3f} "
            f"target={result.target_shadow_loss:.3f} "
            f"shadow_n={result.shadow_size} diff={result.difference_pct:.3f}%"
        )


if __name__ == "__main__":
    main()
