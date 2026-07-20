"""Message-based metric-aware server-side DP strategy for Flower."""

from collections.abc import Iterable, Sequence
from logging import INFO

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp.strategy import (
    DifferentialPrivacyServerSideFixedClipping,
    Strategy,
)
from flwr.supercore.differential_privacy import (
    add_gaussian_noise_inplace,
    compute_stdv,
)


def maximum_pairwise_model_distance(models: Sequence[ArrayRecord]) -> float:
    """Return the maximum mean layer-wise Euclidean distance between models."""
    if len(models) < 2:
        raise ValueError("Metric-aware calibration requires at least two client models.")

    ndarrays = [model.to_numpy_ndarrays() for model in models]
    reference_shapes = [array.shape for array in ndarrays[0]]
    for model_arrays in ndarrays[1:]:
        if [array.shape for array in model_arrays] != reference_shapes:
            raise ValueError("All client models must have matching array shapes.")

    distances: list[float] = []
    for i, model_i in enumerate(ndarrays):
        for model_j in ndarrays[i + 1 :]:
            layer_distances = [
                float(np.linalg.norm((array_i - array_j).ravel(), ord=2))
                for array_i, array_j in zip(model_i, model_j, strict=True)
            ]
            distances.append(float(np.mean(layer_distances)))

    return max(distances)


class MetricPrivacyServerSideFixedClipping(
    DifferentialPrivacyServerSideFixedClipping
):
    """Calibrate server-side Gaussian DP noise by client-model distance.

    This modern Flower wrapper follows the mechanism described in the paper:
    compute the maximum pairwise mean layer distance ``d`` and use
    ``noise_multiplier / d`` for the current round. Client updates are then
    clipped by Flower's server-side fixed-clipping wrapper before aggregation.

    The distance-aware calibration is empirical and does not, by itself,
    establish a formal metric-DP guarantee.
    """

    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_norm: float,
        num_sampled_clients: int,
        arrayrecord_key: str = "arrays",
    ) -> None:
        super().__init__(
            strategy=strategy,
            noise_multiplier=noise_multiplier,
            clipping_norm=clipping_norm,
            num_sampled_clients=num_sampled_clients,
        )
        self.arrayrecord_key = arrayrecord_key
        self.current_distance: float | None = None
        self.current_noise_stdv: float | None = None

    def __repr__(self) -> str:
        """Return a concise strategy description."""
        return "Metric-aware DP Strategy (Server-Side Fixed Clipping)"

    def summary(self) -> None:
        """Log metric-aware DP settings and the wrapped strategy summary."""
        log(INFO, "\t├──> Metric-aware DP settings:")
        log(INFO, "\t│\t├── Base noise multiplier: %s", self.noise_multiplier)
        log(INFO, "\t│\t└── Clipping norm: %s", self.clipping_norm)
        self.strategy.summary()

    def aggregate_train(
        self,
        server_round: int,
        replies: Iterable[Message],
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Measure model divergence, clip, aggregate, and add calibrated noise."""
        reply_list = list(replies)

        # Let Flower's DP wrapper handle and log client errors.
        if any(reply.has_error() for reply in reply_list):
            return super().aggregate_train(server_round, reply_list)

        models: list[ArrayRecord] = []
        for reply in reply_list:
            record = reply.content.get(self.arrayrecord_key)
            if not isinstance(record, ArrayRecord):
                raise ValueError(f"Client reply is missing ArrayRecord {self.arrayrecord_key!r}.")
            models.append(record)

        distance = maximum_pairwise_model_distance(models)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError(
                "The client-model distance must be finite and greater than zero; "
                "noise_multiplier / distance is undefined otherwise."
            )

        self.current_distance = distance
        log(INFO, "aggregate_train: maximum pairwise model distance: %.6f", distance)

        aggregated_arrays, aggregated_metrics = super().aggregate_train(
            server_round, reply_list
        )
        if aggregated_metrics is None:
            aggregated_metrics = MetricRecord()
        aggregated_metrics["metric-dp-distance"] = distance
        if self.current_noise_stdv is not None:
            aggregated_metrics["metric-dp-noise-stdv"] = self.current_noise_stdv
        return aggregated_arrays, aggregated_metrics

    def _add_noise_to_aggregated_arrays(
        self, aggregated_arrays: ArrayRecord
    ) -> ArrayRecord:
        """Add Gaussian noise calibrated with the current model distance."""
        if self.current_distance is None:
            raise RuntimeError("Model distance was not computed before adding noise.")

        calibrated_multiplier = self.noise_multiplier / self.current_distance
        stdv = compute_stdv(
            calibrated_multiplier,
            self.clipping_norm,
            self.num_sampled_clients,
        )
        aggregated_ndarrays = aggregated_arrays.to_numpy_ndarrays()
        add_gaussian_noise_inplace(aggregated_ndarrays, stdv)
        self.current_noise_stdv = stdv

        log(
            INFO,
            "aggregate_train: metric-aware noise with %.6f stdev added",
            stdv,
        )

        return ArrayRecord(
            {
                key: Array(np.asarray(value))
                for key, value in zip(
                    aggregated_arrays.keys(), aggregated_ndarrays, strict=True
                )
            }
        )
