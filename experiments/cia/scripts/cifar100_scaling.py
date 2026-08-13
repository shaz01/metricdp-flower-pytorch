"""Multi-round CIA (IN vs OUT) against the CIFAR-100 100-client/250-round sweep.

Attacks all 6 combos from results/cifar100_scaling/ ({homogeneous, non-iid} x {vanilla,
global-dp, metric-privacy} x fedavg, n=100, r=250) across 3 seeds (42, 43, 44) -- 18 trajectories
per group. A matched IN-remove (target participates,
100 clients) and OUT-remove (target excluded, 99 clients) trajectory per combo, checkpointed at
round 1 and every 10th round through 250. Reuses experiments.cia.attack_runner.run_attack
unmodified -- core CIA code is dataset/class-count-agnostic, no changes needed there.

The finished sweep's own checkpoints are gone (it ran with --delete-model-on-success), so this
always trains fresh trajectories -- CIA needs checkpoints at specific rounds to compute
shadow-vs-test loss, which the accuracy-only sweep never kept around anyway.

Run one group at a time so two can run concurrently on the same GPU without racing on a shared
report file (attack_runner.run_attack is not safe to call twice against the same report_name --
each process holds its own in-memory result snapshot and overwrites the file on every checkpoint):

    uv run python -m experiments.cia.scripts.cifar100_scaling --group in
    uv run python -m experiments.cia.scripts.cifar100_scaling --group out

See experiments/cia/scripts/cifar100_scaling_analysis.py for the round-matched AUC step that
merges cia_in.json and cia_out.json afterward.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from experiments.cia.attack_runner import run_attack
from experiments.cia.datasets.partitions import (
    PartitionViewDataModule,
    in_remove,
    out_remove,
)
from experiments.cia.shadow_dataset import clean_shadow_dataset, noisy_shadow_dataset
from experiments.reproduce.dataset.cifar100 import Cifar100DataModule
from experiments.reproduce.matrix import Combo, Hyperparams
from metricdp_pytorch.utils.device import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "cia_cifar100_scaling"

PARTITION_MODES = ("homogeneous", "non-iid")
PRIVACY_MODES = ("vanilla", "global-dp", "metric-privacy")
AGGREGATION = "fedavg"
NUM_CLIENTS = 100
TARGET_PARTITION_ID = 0
SEEDS = (42, 43, 44)  # matches this repo's standard 3-seed CIA convention (planned_runs.py,
# cifar_chunks.py). Seed 42 was already run as part of the single-seed pilot
# (docs/superpowers/specs/2026-08-09-cia-cifar100-design.md) -- pointing this script at the
# same report files lets run_attack's existing resumability skip those 6 already-complete
# combos per group and only train the 12 new seed-43/44 combos.
NOISE_MULTIPLIER = 0.0182  # same calibration as results/cifar100_scaling/, reused for both
# the 100-client IN trajectory and the 99-client OUT trajectory -- matches existing precedent
# (planned_runs.py's alzheimer-in/out-remove groups do the same across a +-1 client difference).
SHADOW_FRACTION = 0.10
NOISE_STD_FRACTION = 0.20
CHECKPOINT_ROUNDS = (1,) + tuple(range(10, 251, 10))  # first-round signal + every 10th
# round through 250, instead of every round -- bounds peak transient disk to
# 26 * ~18.5MB (this model's state dict) ~= 481MB per trajectory, vs ~4.6GB at every round.

MODEL_MODULE = "experiments.reproduce.cifar100_cnn:create_model"

HYPERPARAMS = Hyperparams(
    clipping_norm=5.0,
    rounds=250,
    local_epochs=5,
    batch_size=32,
    learning_rate=0.001,
    initialization_epochs=20,
    weight_decay=5e-4,
    lr_schedule="none",
)


def _cache_dir(config: Mapping[str, Any]) -> str | None:
    return str(config.get("data-cache-dir", "")).strip() or None


def create_cifar100_in(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Create the 100-client IN-remove view (target participates)."""
    return in_remove(
        Cifar100DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=NUM_CLIENTS,
        target_partition_id=TARGET_PARTITION_ID,
    )


def create_cifar100_out(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Create the 99-client OUT-remove view (target excluded)."""
    return out_remove(
        Cifar100DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=NUM_CLIENTS,
        target_partition_id=TARGET_PARTITION_ID,
    )


def build_combos(group: str) -> list[Combo]:
    """Build the 18 (partition x privacy x seed) trajectories for one group."""
    if group not in ("in", "out"):
        raise ValueError('group must be "in" or "out".')
    data_module = f"experiments.cia.scripts.cifar100_scaling:create_cifar100_{group}"
    active_clients = NUM_CLIENTS if group == "in" else NUM_CLIENTS - 1
    return [
        Combo(
            name_prefix=f"cifar100-{group}-remove",
            num_clients=active_clients,
            partition=partition,
            privacy=privacy,
            aggregation=AGGREGATION,
            seed=seed,
            noise_multiplier=NOISE_MULTIPLIER,
            hyperparams=HYPERPARAMS,
            data_module=data_module,
            model_module=MODEL_MODULE,
        )
        for partition in PARTITION_MODES
        for privacy in PRIVACY_MODES
        for seed in SEEDS
    ]


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("in", "out"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-parallel-clients", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    combos = build_combos(args.group)
    run_attack(
        combos=combos,
        output_dir=output_dir,
        log_path=output_dir / f"progress_{args.group}.log",
        max_parallel_clients=args.max_parallel_clients,
        force=args.force,
        start_message=(
            f"Starting cifar100 {args.group}-remove group: {len(combos)} trajectories"
        ),
        clean_data_module_factory=_clean_shadow,
        noisy_data_module_factory=_noisy_shadow,
        device=resolve_device(),
        checkpoint_rounds=CHECKPOINT_ROUNDS,
        report_name=f"cia_{args.group}.json",
    )


if __name__ == "__main__":
    main()
