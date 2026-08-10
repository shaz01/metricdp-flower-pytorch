"""Ten-client non-IID CIFAR-10 removal-adjacency CIA runs.

Covers both removal views (IN keeps the target, OUT drops it) and all three
privacy modes (vanilla, global-DP, metric-privacy) over the full ten-class
``uoft-cs/cifar10``. Homogeneous partitioning is deliberately excluded here --
it is accuracy-only and lives in
``experiments/client_scaling/scripts/cifar10_homogeneous.py``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.cia.attack_runner import run_attack
from experiments.cia.datasets.partitions import (
    PartitionViewDataModule,
    in_remove,
    out_remove,
)
from experiments.cia.shadow_dataset import clean_shadow_dataset, noisy_shadow_dataset
from experiments.reproduce.dataset.cifar10 import Cifar10DataModule
from experiments.reproduce.matrix import Combo, Hyperparams
from metricdp_pytorch.utils.device import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "cia" / "cifar10_remove"

# Canonical federation size: the IN view runs all 10 clients, the OUT view runs
# the same partitioning minus the target, i.e. 9 active clients.
CANONICAL_NUM_CLIENTS = 10
TARGET_PARTITION_ID = 0

ADJACENCIES = ("in-remove", "out-remove")
PARTITION_MODE = "non-iid"
PRIVACY_MODES = ("vanilla", "global-dp", "metric-privacy")
SEEDS = (42,)

ROUNDS = 20
# Every round: multi-round CIA needs the full loss trajectory, not endpoints.
CHECKPOINT_ROUNDS = tuple(range(1, ROUNDS + 1))
SHADOW_FRACTION = 0.10

NOISE_STD_FRACTION = 0.20
NOISE_MULTIPLIER = 0.01  # TODO: calibrate

# TODO: try increased rounds with lower epochs too
HYPERPARAMS = Hyperparams(
    clipping_norm=5.0,  # TODO: revisit clipping norm for higher clients
    rounds=ROUNDS,
    local_epochs=5,
    batch_size=32,
    learning_rate=0.001,
    initialization_epochs=20,
)


def _cache_dir(config: Mapping[str, Any]) -> str | None:
    return str(config.get("data-cache-dir", "")).strip() or None


def create_in_remove(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """IN view: all ten canonical partitions, target included."""
    return in_remove(
        Cifar10DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=CANONICAL_NUM_CLIENTS,
        target_partition_id=TARGET_PARTITION_ID,
    )


def create_out_remove(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """OUT view: the same ten canonical partitions minus the target."""
    return out_remove(
        Cifar10DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=CANONICAL_NUM_CLIENTS,
        target_partition_id=TARGET_PARTITION_ID,
    )


def _clean_shadow(combo: Combo) -> Any:
    return clean_shadow_dataset(
        combo,
        target_partition_id=TARGET_PARTITION_ID,
        shadow_fraction=SHADOW_FRACTION,
    )


def _noisy_shadow(combo: Combo) -> Any:
    return noisy_shadow_dataset(
        combo,
        target_partition_id=TARGET_PARTITION_ID,
        shadow_fraction=SHADOW_FRACTION,
        std_fraction=NOISE_STD_FRACTION,
    )


def _data_module(adjacency: str) -> str:
    module = "experiments.cia.scripts.cifar10_remove"
    return f"{module}:create_in_remove" if adjacency == "in-remove" else (
        f"{module}:create_out_remove"
    )


def _active_clients(adjacency: str) -> int:
    return (
        CANONICAL_NUM_CLIENTS
        if adjacency == "in-remove"
        else CANONICAL_NUM_CLIENTS - 1
    )


def build_combos(
    *,
    adjacencies: Sequence[str] = ADJACENCIES,
    privacy_modes: Sequence[str] = PRIVACY_MODES,
    seeds: Sequence[int] = SEEDS,
) -> list[Combo]:
    """Build the full adjacency x privacy x seed trajectory list."""
    for adjacency in adjacencies:
        if adjacency not in ADJACENCIES:
            raise ValueError(f"adjacency must come from {ADJACENCIES}.")
    for privacy in privacy_modes:
        if privacy not in PRIVACY_MODES:
            raise ValueError(f"privacy mode must come from {PRIVACY_MODES}.")

    return [
        Combo(
            name_prefix=f"cifar10-{adjacency}",
            num_clients=_active_clients(adjacency),
            partition=PARTITION_MODE,
            privacy=privacy,
            aggregation="fedavg",
            seed=seed,
            noise_multiplier=NOISE_MULTIPLIER,
            hyperparams=HYPERPARAMS,
            data_module=_data_module(adjacency),
            model_module="experiments.reproduce.cifar10_cnn:create_model",
        )
        for adjacency in adjacencies
        for privacy in privacy_modes
        for seed in seeds
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--privacy",
        choices=PRIVACY_MODES,
        required=True,
        help="Exactly one privacy mode per session (keeps Colab chunks small).",
    )
    parser.add_argument(
        "--adjacency",
        choices=ADJACENCIES,
        nargs="+",
        default=list(ADJACENCIES),
        help="Defaults to both views; pass one to split a session further.",
    )
    parser.add_argument("--seed", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--max-parallel-clients",
        type=int,
        default=min(CANONICAL_NUM_CLIENTS, 8),  # TODO: tune per machine.
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combos = build_combos(
        adjacencies=args.adjacency,
        privacy_modes=(args.privacy,),
        seeds=args.seed,
    )
    results = run_attack(
        combos=combos,
        output_dir=output_dir / "runs",
        log_path=output_dir / "runs" / "progress.log",
        max_parallel_clients=args.max_parallel_clients,
        force=args.force,
        start_message=(
            f"CIFAR-10 removal CIA chunk ({args.privacy}): "
            f"{len(combos)} trajectories"
        ),
        clean_data_module_factory=_clean_shadow,
        noisy_data_module_factory=_noisy_shadow,
        device=resolve_device(),
        checkpoint_rounds=CHECKPOINT_ROUNDS,
        report_name="cia.json",
    )
    for result in sorted(results, key=lambda r: (r.run_name, r.server_round)):
        print(
            f"round={result.server_round:2d} {result.partition_mode:12s} "
            f"{result.privacy:15s} clients={result.num_clients:2d} "
            f"agg={result.aggregated_test_loss:.3f} "
            f"clean_diff={result.clean_difference_pct:.3f}% "
            f"noisy_diff={result.noisy_difference_pct:.3f}%"
        )


if __name__ == "__main__":
    main()
