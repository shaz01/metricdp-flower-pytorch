"""Parametrized Alzheimer-MRI removal-adjacency CIA stage runner.

Same shape as ``eurosat_remove.py``/``cifar10_remove.py``: one invocation
trains and attacks both adjacencies for one privacy mode, one partition mode,
at a caller-supplied noise ratio, for the AUC-targeted noise search
(``auc_target_search.py``).

``--clients`` defaults to 48. The Alzheimer-MRI dataset is small (~5,120 train
images across 4 classes, heavily imbalanced -- one class has only 49 images
total), so per-client data is thin at this client count; this is a deliberate
part of what the sweep studies, not an oversight.
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
from experiments.cia.result import CiaResult
from experiments.cia.shadow_dataset import clean_shadow_dataset, noisy_shadow_dataset
from experiments.reproduce.dataset.alzheimer import AlzheimerDataModule
from experiments.reproduce.matrix import Combo, Hyperparams
from metricdp_pytorch.utils.device import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "cia" / "alzheimer_remove"

CANONICAL_NUM_CLIENTS = 48
TARGET_PARTITION_ID = 0

ADJACENCIES = ("in-remove", "out-remove")
PARTITION_MODES = ("homogeneous", "non-iid")
PARTITION_MODE = "non-iid"
PRIVACY_MODES = ("vanilla", "global-dp", "metric-privacy")
SEEDS = (42,)

ROUNDS = 100
CHECKPOINT_ROUNDS = (1,) + tuple(range(10, ROUNDS + 1, 10))
SHADOW_FRACTION = 0.10
NOISE_STD_FRACTION = 0.20

NOISE_MULTIPLIER = 0.00373606409424702  # calibrated 2026-08-17 for this
# model's 6,517,476 params at n=48. Derivation: target noise-to-signal ratio
# ~1, using noise_l2_norm ~= stdv * sqrt(param_count) and
# stdv = noise_multiplier * clipping_norm / num_sampled_clients. Measured
# directly on this model (n=48, homogeneous, global-dp, 3 rounds): signal
# update norm (post-clip, post-aggregation, pre-noise) = 0.9935341638694208,
# giving target_noise_multiplier = signal_norm * 48 / (5.0 * sqrt(6517476))
# = 0.00373606409424702.

HYPERPARAMS = Hyperparams(
    clipping_norm=5.0,
    rounds=ROUNDS,
    local_epochs=5,
    batch_size=32,
    learning_rate=0.001,
    initialization_epochs=20,
)


def _cache_dir(config: Mapping[str, Any]) -> str | None:
    return str(config.get("data-cache-dir", "")).strip() or None


def _canonical_clients(config: Mapping[str, Any], adjacency: str) -> int:
    active_clients = int(config["num-clients"])
    canonical = active_clients if adjacency == "in-remove" else active_clients + 1
    if canonical < 2:
        raise ValueError("Removal adjacency requires at least two canonical clients.")
    return canonical


def create_in_remove(config: Mapping[str, Any]) -> PartitionViewDataModule:
    return in_remove(
        AlzheimerDataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=_canonical_clients(config, "in-remove"),
        target_partition_id=TARGET_PARTITION_ID,
    )


def create_out_remove(config: Mapping[str, Any]) -> PartitionViewDataModule:
    return out_remove(
        AlzheimerDataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=_canonical_clients(config, "out-remove"),
        target_partition_id=TARGET_PARTITION_ID,
    )


def _clean_shadow(combo: Combo) -> Any:
    return clean_shadow_dataset(
        combo, target_partition_id=TARGET_PARTITION_ID, shadow_fraction=SHADOW_FRACTION
    )


def _noisy_shadow(combo: Combo) -> Any:
    return noisy_shadow_dataset(
        combo,
        target_partition_id=TARGET_PARTITION_ID,
        shadow_fraction=SHADOW_FRACTION,
        std_fraction=NOISE_STD_FRACTION,
    )


def _data_module(adjacency: str) -> str:
    module = "experiments.cia.scripts.alzheimer_remove"
    return f"{module}:create_in_remove" if adjacency == "in-remove" else (
        f"{module}:create_out_remove"
    )


def _active_clients(adjacency: str, canonical_num_clients: int) -> int:
    return (
        canonical_num_clients if adjacency == "in-remove" else canonical_num_clients - 1
    )


def build_combos(
    *,
    adjacencies: Sequence[str] = ADJACENCIES,
    privacy_modes: Sequence[str] = PRIVACY_MODES,
    seeds: Sequence[int] = SEEDS,
    canonical_num_clients: int = CANONICAL_NUM_CLIENTS,
    noise_ratio: float | None = None,
    partition: str = PARTITION_MODE,
) -> list[Combo]:
    if canonical_num_clients < 2:
        raise ValueError("canonical_num_clients must be at least 2.")
    for adjacency in adjacencies:
        if adjacency not in ADJACENCIES:
            raise ValueError(f"adjacency must come from {ADJACENCIES}.")
    for privacy in privacy_modes:
        if privacy not in PRIVACY_MODES:
            raise ValueError(f"privacy mode must come from {PRIVACY_MODES}.")
    if partition not in PARTITION_MODES:
        raise ValueError(f"partition must come from {PARTITION_MODES}.")
    if noise_ratio is not None and noise_ratio <= 0:
        raise ValueError("noise_ratio must be positive when provided.")

    return [
        Combo(
            name_prefix=f"alzheimer-{adjacency}",
            num_clients=_active_clients(adjacency, canonical_num_clients),
            partition=partition,
            privacy=privacy,
            aggregation="fedavg",
            seed=seed,
            noise_multiplier=(
                NOISE_MULTIPLIER
                if noise_ratio is None
                else noise_ratio * _active_clients(adjacency, canonical_num_clients)
            ),
            hyperparams=HYPERPARAMS,
            data_module=_data_module(adjacency),
            model_module="experiments.reproduce.paper_cnn:create_model",
        )
        for adjacency in adjacencies
        for privacy in privacy_modes
        for seed in seeds
    ]


def run_stage(
    *,
    privacy: str,
    partition: str,
    seed: int,
    noise_ratio: float | None,
    canonical_num_clients: int,
    output_dir: Path,
    max_parallel_clients: int | None = None,
    force: bool = False,
) -> list[CiaResult]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combos = build_combos(
        privacy_modes=(privacy,),
        seeds=(seed,),
        canonical_num_clients=canonical_num_clients,
        noise_ratio=noise_ratio,
        partition=partition,
    )
    return run_attack(
        combos=combos,
        output_dir=output_dir / "runs",
        log_path=output_dir / "runs" / "progress.log",
        max_parallel_clients=max_parallel_clients or min(canonical_num_clients, 6),
        force=force,
        start_message=(
            f"Alzheimer stage ({privacy}, {partition}, seed={seed}, "
            f"noise_ratio={noise_ratio}): {len(combos)} trajectories"
        ),
        clean_data_module_factory=_clean_shadow,
        noisy_data_module_factory=_noisy_shadow,
        device=resolve_device(),
        checkpoint_rounds=CHECKPOINT_ROUNDS,
        report_name="cia.json",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--privacy", choices=PRIVACY_MODES, required=True)
    parser.add_argument(
        "--adjacency", choices=ADJACENCIES, nargs="+", default=list(ADJACENCIES)
    )
    parser.add_argument("--partition", choices=PARTITION_MODES, default=PARTITION_MODE)
    parser.add_argument("--seed", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--clients", type=int, default=CANONICAL_NUM_CLIENTS)
    parser.add_argument("--noise-ratio", type=float)
    parser.add_argument("--max-parallel-clients", type=int)
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
        canonical_num_clients=args.clients,
        noise_ratio=args.noise_ratio,
        partition=args.partition,
    )
    run_attack(
        combos=combos,
        output_dir=output_dir / "runs",
        log_path=output_dir / "runs" / "progress.log",
        max_parallel_clients=args.max_parallel_clients or min(args.clients, 6),
        force=args.force,
        start_message=(
            f"Alzheimer removal CIA chunk ({args.privacy}, clients={args.clients}, "
            f"noise_ratio={args.noise_ratio}): {len(combos)} trajectories"
        ),
        clean_data_module_factory=_clean_shadow,
        noisy_data_module_factory=_noisy_shadow,
        device=resolve_device(),
        checkpoint_rounds=CHECKPOINT_ROUNDS,
        report_name="cia.json",
    )


if __name__ == "__main__":
    main()
