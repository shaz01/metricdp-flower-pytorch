"""Participant views for IN/OUT client-inference experiments.

The wrapped data module owns a fixed set of *canonical* partitions.  A view
exposes only the partitions participating in one CIA trajectory and remaps
Flower's contiguous active partition IDs to those canonical IDs.

Removal adjacency uses a canonical dataset containing the target::

    IN-remove  = all canonical partitions
    OUT-remove = all canonical partitions except the target

Replacement adjacency uses a canonical dataset containing both the target and
an alternative replacement client::

    IN-replace  = all canonical partitions except the replacement
    OUT-replace = all canonical partitions except the target

Consequently, IN-replace and OUT-replace have the same number of active
clients.  The caller is responsible for constructing disjoint canonical target
and replacement partitions; this module never duplicates or redistributes
samples.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch.utils.data import DataLoader

from metricdp_pytorch.data_module import FederatedDataModule

def in_remove(
        data_module: FederatedDataModule,
        *,
        canonical_num_partitions: int,
        target_partition_id: int,
) -> PartitionViewDataModule:
    """Return the removal-adjacency IN view, including the target."""
    _validate_partition_id(
        "target_partition_id", target_partition_id, canonical_num_partitions
    )
    return _view_excluding(
        data_module,
        canonical_num_partitions=canonical_num_partitions,
        excluded_partition_ids=(),
    )


def out_remove(
        data_module: FederatedDataModule,
        *,
        canonical_num_partitions: int,
        target_partition_id: int,
) -> PartitionViewDataModule:
    """Return the removal-adjacency OUT view, excluding the target."""
    _validate_partition_id(
        "target_partition_id", target_partition_id, canonical_num_partitions
    )
    return _view_excluding(
        data_module,
        canonical_num_partitions=canonical_num_partitions,
        excluded_partition_ids=(target_partition_id,),
    )


def in_replace(
        data_module: FederatedDataModule,
        *,
        canonical_num_partitions: int,
        target_partition_id: int,
        replacement_partition_id: int,
) -> PartitionViewDataModule:
    """Return the replacement-adjacency IN view, selecting the target."""
    _validate_replacement_ids(
        target_partition_id, replacement_partition_id, canonical_num_partitions
    )
    return _view_excluding(
        data_module,
        canonical_num_partitions=canonical_num_partitions,
        excluded_partition_ids=(replacement_partition_id,),
    )


def out_replace(
        data_module: FederatedDataModule,
        *,
        canonical_num_partitions: int,
        target_partition_id: int,
        replacement_partition_id: int,
) -> PartitionViewDataModule:
    """Return the replacement-adjacency OUT view, selecting the replacement."""
    _validate_replacement_ids(
        target_partition_id, replacement_partition_id, canonical_num_partitions
    )
    return _view_excluding(
        data_module,
        canonical_num_partitions=canonical_num_partitions,
        excluded_partition_ids=(target_partition_id,),
    )

##################################################################################
##################################################################################
##################################################################################

class PartitionViewDataModule:
    """Expose selected canonical client partitions as contiguous active IDs."""

    def __init__(
        self,
        data_module: FederatedDataModule,
        *,
        canonical_num_partitions: int,
        active_partition_ids: Sequence[int],
    ) -> None:
        if canonical_num_partitions < 1:
            raise ValueError("canonical_num_partitions must be positive.")

        active_ids = tuple(active_partition_ids)
        if not active_ids:
            raise ValueError("At least one partition must be active.")
        if len(set(active_ids)) != len(active_ids):
            raise ValueError("active_partition_ids must be unique.")
        if any(
            partition_id < 0 or partition_id >= canonical_num_partitions
            for partition_id in active_ids
        ):
            raise ValueError(
                "active_partition_ids must identify canonical partitions."
            )

        self.data_module = data_module
        self.canonical_num_partitions = canonical_num_partitions
        self.active_partition_ids = active_ids

    @property
    def class_names(self) -> Sequence[str]:
        """Expose class names from the wrapped module when available."""
        return getattr(self.data_module, "class_names", ())

    @property
    def num_active_partitions(self) -> int:
        return len(self.active_partition_ids)

    def canonical_partition_id(self, active_partition_id: int) -> int:
        """Translate a contiguous Flower partition ID to its canonical ID."""
        if not 0 <= active_partition_id < self.num_active_partitions:
            raise ValueError(
                f"partition_id must be in [0, {self.num_active_partitions})."
            )
        return self.active_partition_ids[active_partition_id]

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
        dirichlet_alpha: float = 0.5,
        max_samples: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        if num_partitions != self.num_active_partitions:
            raise ValueError(
                "num_partitions must equal the number of active partitions "
                f"({self.num_active_partitions})."
            )
        if client_weights is not None and self.num_active_partitions != self.canonical_num_partitions:
            raise ValueError(
                "client_weights cannot be remapped after excluding a partition; "
                "configure weights on the canonical data module instead."
            )

        canonical_id = self.canonical_partition_id(partition_id)
        return self.data_module.client_loaders(
            canonical_id,
            num_partitions=self.canonical_num_partitions,
            partition_mode=partition_mode,
            batch_size=batch_size,
            seed=seed,
            partition_profile=partition_profile,
            client_weights=client_weights,
            dirichlet_alpha=dirichlet_alpha,
            max_samples=max_samples,
        )

    def server_loaders(
        self,
        *,
        batch_size: int,
        seed: int,
        max_samples: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        return self.data_module.server_loaders(
            batch_size=batch_size,
            seed=seed,
            max_samples=max_samples,
        )


def _validate_partition_id(
    name: str, partition_id: int, canonical_num_partitions: int
) -> None:
    if not 0 <= partition_id < canonical_num_partitions:
        raise ValueError(
            f"{name} must be in [0, {canonical_num_partitions})."
        )


def _view_excluding(
    data_module: FederatedDataModule,
    *,
    canonical_num_partitions: int,
    excluded_partition_ids: Sequence[int],
) -> PartitionViewDataModule:
    excluded = set(excluded_partition_ids)
    return PartitionViewDataModule(
        data_module,
        canonical_num_partitions=canonical_num_partitions,
        active_partition_ids=tuple(
            partition_id
            for partition_id in range(canonical_num_partitions)
            if partition_id not in excluded
        ),
    )



def _validate_replacement_ids(
    target_partition_id: int,
    replacement_partition_id: int,
    canonical_num_partitions: int,
) -> None:
    _validate_partition_id(
        "target_partition_id", target_partition_id, canonical_num_partitions
    )
    _validate_partition_id(
        "replacement_partition_id",
        replacement_partition_id,
        canonical_num_partitions,
    )
    if target_partition_id == replacement_partition_id:
        raise ValueError(
            "target_partition_id and replacement_partition_id must differ."
        )
