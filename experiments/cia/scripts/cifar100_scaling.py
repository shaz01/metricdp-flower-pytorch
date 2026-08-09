"""Multi-round CIA (IN vs OUT) against the CIFAR-100 100-client/250-round sweep.

Attacks all 6 combos from results/cifar100_scaling/ ({homogeneous, non-iid} x {vanilla,
global-dp, metric-privacy} x fedavg, n=100, r=250): a matched IN-remove (target participates,
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
SEED = 42
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


if __name__ == "__main__":
    raise SystemExit("cifar100_scaling.py's CLI is added in Task 2.")
