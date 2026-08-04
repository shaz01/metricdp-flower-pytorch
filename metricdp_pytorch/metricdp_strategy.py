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

from metricdp_pytorch.dp_diagnostics import (
    add_diagnostics,
    clipping_diagnostics,
    signal_to_noise_diagnostics,
)


def pairwise_model_distances(models: Sequence[ArrayRecord]) -> list[float]:
    """Return upper-triangle mean layer-wise distances in row-major order."""
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
    return distances


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
        self.current_round_diagnostics: dict[str, float | int] = {}

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

        client_models: list[tuple[int, ArrayRecord]] = []
        for fallback_id, reply in enumerate(reply_list):
            record = reply.content.get(self.arrayrecord_key)
            if not isinstance(record, ArrayRecord):
                raise ValueError(f"Client reply is missing ArrayRecord {self.arrayrecord_key!r}.")
            metrics = next(iter(reply.content.metric_records.values()), MetricRecord())
            client_id = int(metrics.get("client-id", fallback_id))
            client_models.append((client_id, record))
        client_models.sort(key=lambda item: item[0])
        client_ids = [client_id for client_id, _ in client_models]
        models = [model for _, model in client_models]

        distances = pairwise_model_distances(models)
        pair_client_i = [
            client_ids[i]
            for i in range(len(client_ids))
            for _j in range(i + 1, len(client_ids))
        ]
        pair_client_j = [
            client_ids[j]
            for i in range(len(client_ids))
            for j in range(i + 1, len(client_ids))
        ]
        distance = max(distances)
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError(
                "The client-model distance must be finite and greater than zero; "
                "noise_multiplier / distance is undefined otherwise."
            )

        self.current_distance = distance
        self.current_noise_stdv = None
        self.current_round_diagnostics = {}
        log(INFO, "aggregate_train: maximum pairwise model distance: %.6f", distance)
        diagnostics = clipping_diagnostics(
            reply_list,
            current_arrays=self.current_arrays,
            clipping_norm=self.clipping_norm,
            arrayrecord_key=self.arrayrecord_key,
        )
        diagnostics.update(
            {
                "metric-dp-pairwise-distances": distances,
                "metric-dp-pairwise-client-i": pair_client_i,
                "metric-dp-pairwise-client-j": pair_client_j,
                "metric-dp-distance-min": float(np.min(distances)),
                "metric-dp-distance-median": float(np.median(distances)),
                "metric-dp-distance-mean": float(np.mean(distances)),
                "metric-dp-distance": distance,
            }
        )

        aggregated_arrays, aggregated_metrics = super().aggregate_train(
            server_round, reply_list
        )
        diagnostics.update(self.current_round_diagnostics)
        if self.current_noise_stdv is not None:
            diagnostics["metric-dp-noise-stdv"] = self.current_noise_stdv
        return aggregated_arrays, add_diagnostics(aggregated_metrics, diagnostics)

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
        self.current_round_diagnostics = signal_to_noise_diagnostics(
            aggregated_arrays,
            current_arrays=self.current_arrays,
            noise_stdv=stdv,
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
