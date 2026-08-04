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
    data_module: str
    model_module: str

    def list_combos(self, *, name_prefix: str, num_clients: int) -> list[Combo]:
        """List every meaningful combo in this matrix.

        Vanilla training does not use the noise multiplier, so only the first
        configured value is retained for that privacy mode.
        """

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
                data_module=self.data_module,
                model_module=self.model_module,
            )
            for partition, privacy, aggregation, seed in product(
                self.partitions,
                self.privacy_modes,
                self.aggregations,
                self.seeds,
            )
            for noise_multiplier in (
                self.noise_multipliers[:1]
                if privacy == "vanilla"
                else self.noise_multipliers
            )
        ]
