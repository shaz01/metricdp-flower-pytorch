"""Pluggable data-module interface used by Flower clients and servers."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from torch.utils.data import DataLoader


@runtime_checkable
class FederatedDataModule(Protocol):
    """Provide client and server loaders independently of dataset storage."""

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
        """Return one client's local train and held-out loaders."""

    def server_loaders(
        self,
        *,
        batch_size: int,
        seed: int,
        max_samples: int = 0,
    ) -> tuple[DataLoader, DataLoader]:
        """Return server validation and final-test loaders."""


def load_data_module(
    factory_path: str,
    config: Mapping[str, Any],
) -> FederatedDataModule:
    """Load ``module:factory`` and construct a validated data module.

    A new dataset can be plugged in by implementing ``FederatedDataModule`` and
    exposing a factory that accepts the run configuration mapping.
    """
    module_name, separator, factory_name = factory_path.partition(":")
    if not separator or not module_name or not factory_name:
        raise ValueError("data-module must use the format 'package.module:factory'.")
    module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise TypeError(f"Data-module factory {factory_path!r} is not callable.")
    data_module = factory(config)
    if not isinstance(data_module, FederatedDataModule):
        raise TypeError(
            f"Factory {factory_path!r} did not return a FederatedDataModule."
        )
    return data_module
