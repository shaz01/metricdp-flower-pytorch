"""Matrix and hyperparameter configuration for reproduction runs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from experiments.reproduce.matrix.hyperparams import Hyperparams
from experiments.reproduce.matrix.combo import Combo


@dataclass(frozen=True)
class Matrix:
    """Cartesian dimensions and shared hyperparameters for reproduction runs."""

    partitions: tuple[str, ...]
    privacy_modes: tuple[str, ...]
    aggregations: tuple[str, ...]
    seeds: tuple[int, ...]
    noise_multipliers: tuple[float, ...]
    hyperparams: Hyperparams

    def list_combos(self, *, name_prefix: str, num_clients: int) -> list[Combo]:
        """List one combo for every point in this matrix."""

        return [
            Combo(
                name_prefix=name_prefix,
                num_clients=num_clients,
                partition=partition,
                privacy=privacy,
                aggregation=aggregation,
                seed=seed,
                noise_multiplier=noise_multiplier,
                hyperparams=self.hyperparams,
            )
            for partition, privacy, aggregation, seed, noise_multiplier in product(
                self.partitions,
                self.privacy_modes,
                self.aggregations,
                self.seeds,
                self.noise_multipliers,
            )
        ]
