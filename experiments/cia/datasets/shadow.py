"""Dataset-independent shadow-data-module decorator for CIA experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from torch.utils.data import DataLoader, Dataset, Subset

from metricdp_pytorch.data_module import FederatedDataModule, load_data_module
from metricdp_pytorch.utils.data import (
    RecordImageDataset,
    labels_from_records,
    make_indexed_loader,
)
from metricdp_pytorch.utils.split_data import split_stratified

DEFAULT_BASE_DATA_MODULE = (
    "experiments.reproduce.dataset.alzheimer:create_data_module"
)


def _client_weights(value: Any) -> tuple[float, ...] | None:
    if not value:
        return None
    if isinstance(value, str):
        return tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not isinstance(value, (list, tuple)):
        raise TypeError("client-weights must be a sequence or comma-separated string.")
    return tuple(float(weight) for weight in value)


def _labels_from_dataset(dataset: Dataset) -> list[int]:
    """Read labels through common metadata or the ``(sample, label)`` contract."""
    if isinstance(dataset, Subset):
        parent_labels = _labels_from_dataset(dataset.dataset)
        return [parent_labels[index] for index in dataset.indices]
    if isinstance(dataset, RecordImageDataset):
        return labels_from_records(
            dataset.dataset, label_column=dataset.label_column
        ).tolist()
    for attribute in ("targets", "labels"):
        values = getattr(dataset, attribute, None)
        if values is not None and len(values) == len(dataset):
            return [int(value) for value in values]

    labels: list[int] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise TypeError(
                "A shadow data module requires client datasets yielding "
                "(sample, label) pairs."
            )
        labels.append(int(sample[1]))
    return labels


class ShadowDataModule:
    """Add a target shadow loader to any federated data module.

    Normal client/server loading is delegated unchanged. To construct the
    shadow set, the decorator asks the wrapped module for the target client's
    actual training loader and takes a deterministic stratified subset of that
    loader's dataset. Consequently it does not need to know how the wrapped
    module stores, partitions, caps, or transforms its data.
    """

    def __init__(
        self,
        data_module: FederatedDataModule,
        *,
        num_clients: int,
        target_partition_id: int,
        shadow_fraction: float = 0.10,
        partition_mode: str = "homogeneous",
        partition_profile: str = "auto",
        client_weights: Sequence[float] | None = None,
    ) -> None:
        if num_clients < 1:
            raise ValueError("num_clients must be positive.")
        if not 0 <= target_partition_id < num_clients:
            raise ValueError(f"target_partition_id must be in [0, {num_clients}).")
        if not 0.0 < shadow_fraction < 1.0:
            raise ValueError("shadow_fraction must be in (0, 1).")
        if client_weights is not None and len(client_weights) != num_clients:
            raise ValueError("client_weights must contain one value per client.")

        self.data_module = data_module
        self.num_clients = num_clients
        self.target_partition_id = target_partition_id
        self.shadow_fraction = shadow_fraction
        self.partition_mode = partition_mode
        self.partition_profile = partition_profile
        self.client_weights = client_weights

    @property
    def class_names(self) -> Sequence[str]:
        """Expose class names from the wrapped data module when available."""
        return getattr(self.data_module, "class_names", ())

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
        return self.data_module.client_loaders(
            partition_id,
            num_partitions=num_partitions,
            partition_mode=partition_mode,
            batch_size=batch_size,
            seed=seed,
            partition_profile=partition_profile,
            client_weights=client_weights,
            max_samples=max_samples,
        )

    def server_loaders(
        self, *, batch_size: int, seed: int, max_samples: int = 0
    ) -> tuple[DataLoader, DataLoader]:
        return self.data_module.server_loaders(
            batch_size=batch_size,
            seed=seed,
            max_samples=max_samples,
        )

    def target_shadow_loader(
        self, *, batch_size: int, seed: int, max_samples: int = 0
    ) -> DataLoader:
        """Return a stratified subset of the target's actual training data."""
        target_train_loader, _ = self.data_module.client_loaders(
            self.target_partition_id,
            num_partitions=self.num_clients,
            partition_mode=self.partition_mode,
            batch_size=batch_size,
            seed=seed,
            partition_profile=self.partition_profile,
            client_weights=self.client_weights,
            max_samples=max_samples,
        )
        target_train_dataset = target_train_loader.dataset
        labels = _labels_from_dataset(target_train_dataset)
        shadow_indices, _rest = split_stratified(
            labels,
            list(range(len(target_train_dataset))),
            self.shadow_fraction,
            seed=seed,
        )
        return make_indexed_loader(
            target_train_dataset,
            shadow_indices,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )


def create_shadow_data_module(config: Mapping[str, Any]) -> ShadowDataModule:
    """Wrap a configured data-module factory with shadow-set behaviour.

    ``shadow-base-data-module`` names the wrapped ``module:factory`` and
    defaults to the standard scalable Alzheimer module.
    """
    factory_path = str(
        config.get("shadow-base-data-module", DEFAULT_BASE_DATA_MODULE)
    )
    base_data_module = load_data_module(factory_path, config)
    return ShadowDataModule(
        base_data_module,
        num_clients=int(config.get("num-clients", 4)),
        target_partition_id=int(config.get("target-partition-id", 0)),
        shadow_fraction=float(config.get("shadow-fraction", 0.10)),
        partition_mode=str(config.get("partition-mode", "homogeneous")),
        partition_profile=str(config.get("partition-profile", "auto")),
        client_weights=_client_weights(config.get("client-weights")),
    )
