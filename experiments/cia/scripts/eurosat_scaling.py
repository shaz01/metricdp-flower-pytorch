"""Multi-round CIA (IN vs OUT) against the EuroSAT 48-client/100-round sweep.

Attacks all 6 combos from results/eurosat_scaling/ ({homogeneous, non-iid} x {vanilla,
global-dp, metric-privacy} x fedavg, n=48, r=100) across 3 seeds (42, 43, 44) -- 18 trajectories
per group. A matched IN-remove (target participates, 48 clients) and OUT-remove (target
excluded, 47 clients) trajectory per combo, checkpointed at round 1 and every 10th round through
100. Reuses experiments.cia.attack_runner.run_attack unmodified -- core CIA code is
dataset/class-count-agnostic, no changes needed there (verified byte-identical to
feature/cifar100-scaling's copies of attack_runner.py, datasets/partitions.py, and
shadow_dataset.py during design).

Going straight to 3 seeds rather than a single-seed pilot: CIFAR-100 CIA started with seed 42
only and that turned out statistically underpowered (see
docs/superpowers/specs/2026-08-09-cia-cifar100-design.md and the subsequent multi-seed rerun),
so this experiment skips repeating that mistake.

noise_multiplier is reused directly from experiments/eurosat_scaling/sweep_eurosat_scaling.py's
own empirical calibration -- no new calibration step, exactly mirroring how CIFAR-100 CIA reused
its own accuracy sweep's calibrated value rather than recalibrating for CIA specifically.

No weight_decay/lr_schedule: this branch is based on master, whose Hyperparams dataclass has
neither field (feature/cifar100-scaling added them independently; this branch deliberately
doesn't duplicate that, same decision already made for sweep_eurosat_scaling.py).

Run one group at a time per attack_runner.run_attack's own not-safe-to-call-twice-against-the-
same-report-file constraint:

    uv run python -m experiments.cia.scripts.eurosat_scaling --group in
    uv run python -m experiments.cia.scripts.eurosat_scaling --group out
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
from experiments.reproduce.dataset.eurosat import EurosatDataModule
from experiments.reproduce.matrix import Combo, Hyperparams
from metricdp_pytorch.utils.device import resolve_device

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "cia_eurosat_scaling"

PARTITION_MODES = ("homogeneous", "non-iid")
PRIVACY_MODES = ("vanilla", "global-dp", "metric-privacy")
AGGREGATION = "fedavg"
NUM_CLIENTS = 48
TARGET_PARTITION_ID = 0
SEEDS = (42, 43, 44)  # matches this repo's standard 3-seed CIA convention, and CIFAR-100 CIA's
# own eventual 3-seed rerun -- going straight to 3 here rather than a single-seed pilot.
NOISE_MULTIPLIER = 0.03710712210729851  # reused unmodified from
# experiments/eurosat_scaling/sweep_eurosat_scaling.py's own empirical calibration.
SHADOW_FRACTION = 0.10
NOISE_STD_FRACTION = 0.20
CHECKPOINT_ROUNDS = (1,) + tuple(range(10, 101, 10))  # first-round signal + every 10th
# round through 100, scaled down from CIFAR-100 CIA's every-10th-through-250 to match this
# sweep's shorter round budget.

MODEL_MODULE = "experiments.reproduce.eurosat_cnn:create_model"

HYPERPARAMS = Hyperparams(
    clipping_norm=5.0,
    rounds=100,
    local_epochs=5,
    batch_size=32,
    learning_rate=0.001,
    initialization_epochs=20,
)


def _cache_dir(config: Mapping[str, Any]) -> str | None:
    return str(config.get("data-cache-dir", "")).strip() or None


def create_eurosat_in(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Create the 48-client IN-remove view (target participates)."""
    return in_remove(
        EurosatDataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=NUM_CLIENTS,
        target_partition_id=TARGET_PARTITION_ID,
    )


def create_eurosat_out(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """Create the 47-client OUT-remove view (target excluded)."""
    return out_remove(
        EurosatDataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=NUM_CLIENTS,
        target_partition_id=TARGET_PARTITION_ID,
    )


def build_combos(group: str) -> list[Combo]:
    """Build the 18 (partition x privacy x seed) trajectories for one group."""
    if group not in ("in", "out"):
        raise ValueError('group must be "in" or "out".')
    data_module = f"experiments.cia.scripts.eurosat_scaling:create_eurosat_{group}"
    active_clients = NUM_CLIENTS if group == "in" else NUM_CLIENTS - 1
    return [
        Combo(
            name_prefix=f"eurosat-{group}-remove",
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
    parser.add_argument("--max-parallel-clients", type=int, default=6)
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
            f"Starting eurosat {args.group}-remove group: {len(combos)} trajectories"
        ),
        clean_data_module_factory=_clean_shadow,
        noisy_data_module_factory=_noisy_shadow,
        device=resolve_device(),
        checkpoint_rounds=CHECKPOINT_ROUNDS,
        report_name=f"cia_{args.group}.json",
    )


if __name__ == "__main__":
    main()
