"""Paper-exact Alzheimer data and shadow module for first-round CIA.

Table 9 as published gives Client 2 as Total=1591, Class0=180. Those values
conflict with both its row sum and Table 1's train totals. Substituting
Total=1491 and Class0=80 reconciles every row, column, and grand-total check;
the corrected values are used below.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from experiments.cia.datasets.shadow import ShadowDataModule
from experiments.reproduce.dataset.alzheimer import (
    AlzheimerDataModule,
    AlzheimerMRIDataset,
)
from metricdp_pytorch.utils.data import labels_from_records, make_client_loaders
from metricdp_pytorch.utils.split_data import partition_by_class_counts

PAPER_CIA_CLIENT_COUNTS = (
    (120, 9, 1122, 496),  # Client 1 -- attacker,  total 1747
    (80, 11, 894, 506),   # Client 2 -- bystander, total 1491 (corrected)
    (524, 29, 550, 779),  # Client 3 -- target,    total 1882
)
PAPER_CIA_NUM_CLIENTS = len(PAPER_CIA_CLIENT_COUNTS)
PAPER_CIA_TARGET_PARTITION_ID = 2
PAPER_CIA_SHADOW_FRACTION = 0.10
PAPER_CIA_TRAIN_FRACTION = 0.8


class PaperCiaDataModule(AlzheimerDataModule):
    """Alzheimer module using the corrected, exact Table 9 client counts."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        train_fraction: float = PAPER_CIA_TRAIN_FRACTION,
    ) -> None:
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1).")
        super().__init__(cache_dir)
        self.train_fraction = train_fraction

    def partitions(self, seed: int) -> list[list[int]]:
        split = self.dataset["train"]
        return partition_by_class_counts(
            labels_from_records(split), PAPER_CIA_CLIENT_COUNTS, seed=seed
        )

    def client_loaders(
        self,
        partition_id: int,
        *,
        num_partitions: int,
        partition_mode: str,
        batch_size: int,
        seed: int,
        partition_profile: str = "auto",
        client_weights: Sequence[float] | None = None,
        max_samples: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        del partition_mode, partition_profile
        if num_partitions != PAPER_CIA_NUM_CLIENTS:
            raise ValueError("The exact paper CIA distribution requires three clients.")
        if client_weights is not None:
            raise ValueError("The exact paper CIA distribution does not use client weights.")
        partitions = self.partitions(seed)
        if not 0 <= partition_id < len(partitions):
            raise ValueError(
                f"partition_id must be in [0, {PAPER_CIA_NUM_CLIENTS})."
            )
        split = self.dataset["train"]
        return make_client_loaders(
            AlzheimerMRIDataset(split),
            labels_from_records(split),
            partitions[partition_id],
            batch_size=batch_size,
            seed=seed + partition_id,
            train_fraction=self.train_fraction,
            max_samples=max_samples,
        )


class PaperShadowDataModule(ShadowDataModule):
    """Paper-exact CIA data module composed with generic shadow behaviour."""

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        target_partition_id: int = PAPER_CIA_TARGET_PARTITION_ID,
        shadow_fraction: float = PAPER_CIA_SHADOW_FRACTION,
        train_fraction: float = PAPER_CIA_TRAIN_FRACTION,
    ) -> None:
        paper_data_module = PaperCiaDataModule(
            cache_dir=cache_dir, train_fraction=train_fraction
        )
        super().__init__(
            paper_data_module,
            num_clients=PAPER_CIA_NUM_CLIENTS,
            target_partition_id=target_partition_id,
            shadow_fraction=shadow_fraction,
            partition_mode="homogeneous",
            partition_profile="exact",
        )

    @property
    def paper_data_module(self) -> PaperCiaDataModule:
        return self.data_module  # type: ignore[return-value]


def create_paper_shadow_data_module(
    config: Mapping[str, Any],
) -> PaperShadowDataModule:
    """Build the paper-exact shadow module from Flower run configuration."""
    num_clients = int(config.get("num-clients", PAPER_CIA_NUM_CLIENTS))
    if num_clients != PAPER_CIA_NUM_CLIENTS:
        raise ValueError("The exact paper CIA distribution requires three clients.")
    cache_dir = str(config.get("data-cache-dir", "")).strip() or None
    return PaperShadowDataModule(
        cache_dir=cache_dir,
        target_partition_id=int(
            config.get("target-partition-id", PAPER_CIA_TARGET_PARTITION_ID)
        ),
        shadow_fraction=float(
            config.get("shadow-fraction", PAPER_CIA_SHADOW_FRACTION)
        ),
        train_fraction=float(config.get("train-fraction", PAPER_CIA_TRAIN_FRACTION)),
    )
