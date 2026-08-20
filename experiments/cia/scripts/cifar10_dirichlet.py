"""Dirichlet CIFAR-10 removal-adjacency CIA runs.

This entry point permits only Dirichlet partitioning. The concentration must
be supplied explicitly with ``--dirichlet-alpha``; there is intentionally no
alpha default. It covers both removal views (IN keeps the target, OUT drops
it) and all three privacy modes over the full ten-class ``uoft-cs/cifar10``.

``--clients`` sets the *canonical* federation size, i.e. the partition count the
dataset is split into. The IN view runs all of them; the OUT view runs the same
partitioning minus the target, so it trains one client fewer. Both views keep
the same canonical partitioning, which is what makes the target's shadow split
identical across a matched IN/OUT pair.
"""

from __future__ import annotations

import argparse
import math
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
RESULTS_ROOT = PROJECT_ROOT / "results" / "dirichlet" / "cifar10"

# Default canonical federation size; override with --clients. The IN view runs
# all canonical clients, the OUT view runs the same partitioning minus the
# target, i.e. one client fewer.
CANONICAL_NUM_CLIENTS = 10
TARGET_PARTITION_ID = 0

ADJACENCIES = ("in-remove", "out-remove")
PARTITION_MODE = "dirichlet"
PRIVACY_MODES = ("vanilla", "global-dp", "metric-privacy")
SEEDS = (42,)

ROUNDS = 20
# Every round: multi-round CIA needs the full loss trajectory, not endpoints.
CHECKPOINT_ROUNDS = tuple(range(1, ROUNDS + 1))
SHADOW_FRACTION = 0.10

NOISE_STD_FRACTION = 0.20
# Retained as the historical default.  A fixed ratio instead uses
# ``ratio * active_clients`` separately for the IN and OUT views, preserving
# the global-DP noise scale despite removal reducing the OUT client count.
NOISE_MULTIPLIER = 0.0182

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


def _canonical_clients(config: Mapping[str, Any], adjacency: str) -> int:
    """Recover the canonical partition count from the active client count.

    ``config['num-clients']`` is the number of clients actually training, which
    is the canonical count for the IN view and one less for the OUT view.
    Deriving it here keeps both views on one canonical partitioning without
    threading a second client-count argument through the runner.
    """
    active_clients = int(config["num-clients"])
    canonical = active_clients if adjacency == "in-remove" else active_clients + 1
    if canonical < 2:
        raise ValueError("Removal adjacency requires at least two canonical clients.")
    return canonical


def create_in_remove(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """IN view: every canonical partition, target included."""
    return in_remove(
        Cifar10DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=_canonical_clients(config, "in-remove"),
        target_partition_id=TARGET_PARTITION_ID,
    )


def create_out_remove(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """OUT view: the same canonical partitions minus the target."""
    return out_remove(
        Cifar10DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=_canonical_clients(config, "out-remove"),
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
    module = "experiments.cia.scripts.cifar10_dirichlet"
    return f"{module}:create_in_remove" if adjacency == "in-remove" else (
        f"{module}:create_out_remove"
    )


def _active_clients(adjacency: str, canonical_num_clients: int) -> int:
    return (
        canonical_num_clients
        if adjacency == "in-remove"
        else canonical_num_clients - 1
    )


def build_combos(
    *,
    adjacencies: Sequence[str] = ADJACENCIES,
    privacy_modes: Sequence[str] = PRIVACY_MODES,
    seeds: Sequence[int] = SEEDS,
    dirichlet_alpha: float,
    canonical_num_clients: int = CANONICAL_NUM_CLIENTS,
    noise_ratio: float | None = None,
) -> list[Combo]:
    """Build the full adjacency x privacy x seed trajectory list."""
    if canonical_num_clients < 2:
        raise ValueError("canonical_num_clients must be at least 2.")
    if not math.isfinite(dirichlet_alpha) or dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be finite and positive.")
    for adjacency in adjacencies:
        if adjacency not in ADJACENCIES:
            raise ValueError(f"adjacency must come from {ADJACENCIES}.")
    for privacy in privacy_modes:
        if privacy not in PRIVACY_MODES:
            raise ValueError(f"privacy mode must come from {PRIVACY_MODES}.")
    if noise_ratio is not None and noise_ratio <= 0:
        raise ValueError("noise_ratio must be positive when provided.")

    return [
        Combo(
            name_prefix=f"cifar10-{adjacency}",
            num_clients=_active_clients(adjacency, canonical_num_clients),
            partition=PARTITION_MODE,
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
            model_module="experiments.reproduce.cifar10_cnn:create_model",
            dirichlet_alpha=dirichlet_alpha,
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
        "--dirichlet-alpha",
        type=float,
        required=True,
        help="Positive Dirichlet concentration; no default is provided.",
    )
    parser.add_argument(
        "--clients",
        type=int,
        default=CANONICAL_NUM_CLIENTS,
        help=(
            "Canonical federation size; the OUT view trains one client fewer "
            f"(default: {CANONICAL_NUM_CLIENTS})."
        ),
    )
    parser.add_argument(
        "--noise-ratio",
        type=float,
        help=(
            "Fixed noise ratio: each view uses ratio times its active client "
            "count (the historical calibrated multiplier is the default)."
        ),
    )
    parser.add_argument(
        "--max-parallel-clients",
        type=int,
        help="Defaults to min(clients, 8).",  # TODO: tune per machine.
    )
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_dir = (RESULTS_ROOT / f"{args.clients}_clients").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combos = build_combos(
        adjacencies=args.adjacency,
        privacy_modes=(args.privacy,),
        seeds=args.seed,
        dirichlet_alpha=args.dirichlet_alpha,
        canonical_num_clients=args.clients,
        noise_ratio=args.noise_ratio,
    )
    results = run_attack(
        combos=combos,
        output_dir=output_dir,
        log_path=output_dir / "progress.log",
        max_parallel_clients=args.max_parallel_clients or min(args.clients, 8),
        force=args.force,
        start_message=(
            f"CIFAR-10 removal CIA chunk ({args.privacy}, "
            f"clients={args.clients}, alpha={args.dirichlet_alpha}, "
            f"noise_ratio={args.noise_ratio}): "
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
