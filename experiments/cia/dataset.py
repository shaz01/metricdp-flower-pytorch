"""3-client Alzheimer-MRI partition for the paper's first-round CIA experiment.

Table 9 as published gives Client 2 as Total=1591, Class0=180 -- internally
row-consistent (180+11+894+506=1591) but its Class-0 column then totals 824
and the grand total is 5220, both exceeding Table 1's known train counts
(Class-0 total 724, grand total 5120). Substituting Total=1491, Class0=80
reconciles every check: the row sum, the column sums against Table 1
((120+80+524, 9+11+29, 1122+894+550, 496+506+779) == (724, 49, 2566, 1781)),
and the grand total (1747+1491+1882 == 5120). This is treated as a published
typo and the corrected values are used here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.utils.data import DataLoader

from experiments.reproduce.dataset.alzheimer import (
    AlzheimerMRIDataset,
    load_alzheimer_dataset,
)
from metricdp_pytorch.utils.data import (
    cap_indices,
    labels_from_records,
    make_client_loaders,
    make_indexed_loader,
    make_server_loaders,
)
from metricdp_pytorch.utils.split_data import partition_by_class_counts, split_stratified

# Rows are clients 1-3 (attacker, bystander, target); columns are sorted
# unique labels 0-3. See module docstring for the Table 9 correction.
CIA_CLIENT_COUNTS = (
    (120, 9, 1122, 496),  # Client 1 -- attacker,  total 1747
    (80, 11, 894, 506),   # Client 2 -- bystander, total 1491 (corrected)
    (524, 29, 550, 779),  # Client 3 -- target,    total 1882
)
CIA_NUM_CLIENTS = len(CIA_CLIENT_COUNTS)
CIA_TARGET_PARTITION_ID = 2
CIA_SHADOW_FRACTION = 0.10
CIA_TRAIN_FRACTION = 0.8


class CiaDataModule:
    """3-client Alzheimer partition matching the paper's CIA experiment (Table 9)."""

    def __init__(self, cache_dir: str | None = None) -> None:
        self.cache_dir = cache_dir
        self._dataset: Any = None

    @property
    def dataset(self) -> Any:
        if self._dataset is None:
            self._dataset = load_alzheimer_dataset(self.cache_dir)
        return self._dataset

    def _train_split_and_labels(self):
        split = self.dataset["train"]
        return split, labels_from_records(split)

    def partitions(self, seed: int) -> list[list[int]]:
        _, labels = self._train_split_and_labels()
        return partition_by_class_counts(labels, CIA_CLIENT_COUNTS, seed=seed)

    def client_loaders(
        self,
        partition_id: int,
        *,
        batch_size: int,
        seed: int,
        max_samples: int = 0,
        **_ignored: Any,
    ) -> tuple[DataLoader, DataLoader]:
        """Ignores partition_mode/num_partitions/partition_profile/client_weights;
        this module always uses the fixed Table 9 (corrected) partition."""
        split, labels = self._train_split_and_labels()
        partitions = self.partitions(seed)
        if not 0 <= partition_id < len(partitions):
            raise ValueError(f"partition_id must be in [0, {CIA_NUM_CLIENTS}).")
        return make_client_loaders(
            AlzheimerMRIDataset(split),
            labels,
            partitions[partition_id],
            batch_size=batch_size,
            seed=seed + partition_id,
            train_fraction=CIA_TRAIN_FRACTION,
            max_samples=max_samples,
        )

    def target_shadow_loader(
        self, *, batch_size: int, seed: int, max_samples: int = 0
    ) -> DataLoader:
        """The attacker's shadow set: a stratified 10% of the target's train
        indices (Section 7.4.1). This overlaps with, and is not excluded
        from, what the target actually trains on."""
        split, labels = self._train_split_and_labels()
        target_indices = cap_indices(
            self.partitions(seed)[CIA_TARGET_PARTITION_ID], max_samples
        )
        train_indices, _test_indices = split_stratified(
            labels,
            target_indices,
            CIA_TRAIN_FRACTION,
            seed=seed + CIA_TARGET_PARTITION_ID,
        )
        shadow_indices, _rest = split_stratified(
            labels, train_indices, CIA_SHADOW_FRACTION, seed=seed
        )
        return make_indexed_loader(
            AlzheimerMRIDataset(split),
            shadow_indices,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )

    def server_loaders(
        self, *, batch_size: int, seed: int, max_samples: int = 0
    ) -> tuple[DataLoader, DataLoader]:
        split = self.dataset["test"]
        return make_server_loaders(
            AlzheimerMRIDataset(split),
            labels_from_records(split),
            batch_size=batch_size,
            seed=seed,
            validation_fraction=0.5,
            max_samples=max_samples,
        )


def create_cia_data_module(config: Mapping[str, Any]) -> CiaDataModule:
    """Factory used by the configurable ClientApp and ServerApp."""
    cache_dir = str(config.get("data-cache-dir", "")).strip() or None
    return CiaDataModule(cache_dir)
