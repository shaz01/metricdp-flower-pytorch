"""Diagnostics shared by the global- and metric-DP strategy wrappers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from logging import INFO

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy


def model_update_l2_norm(model: ArrayRecord, reference: ArrayRecord) -> float:
    """Return the full-vector L2 norm of ``model - reference``."""
    model_arrays = model.to_numpy_ndarrays()
    reference_arrays = reference.to_numpy_ndarrays()
    if len(model_arrays) != len(reference_arrays) or any(
        left.shape != right.shape
        for left, right in zip(model_arrays, reference_arrays, strict=True)
    ):
        raise ValueError("Model and reference arrays must have matching shapes.")
    squared_norm = sum(
        float(np.sum(np.square(left - right, dtype=np.float64)))
        for left, right in zip(model_arrays, reference_arrays, strict=True)
    )
    return float(np.sqrt(squared_norm))


def parameter_count(arrays: ArrayRecord) -> int:
    """Return the number of scalar model parameters."""
    return sum(array.size for array in arrays.to_numpy_ndarrays())


def clipping_diagnostics(
    replies: Sequence[Message],
    *,
    current_arrays: ArrayRecord,
    clipping_norm: float,
    arrayrecord_key: str = "arrays",
) -> dict[str, float | int | list[float] | list[int]]:
    """Measure client update norms before Flower clips the replies."""
    rows: list[tuple[int, float]] = []
    for fallback_id, reply in enumerate(replies):
        # A reply from a client that errored (e.g. a CUDA OOM mid-training) carries
        # no content -- .content raises ValueError in that case instead of returning
        # None, which used to crash the whole server process on any client failure
        # during a global-dp round. Skip it exactly like the non-ArrayRecord case
        # below already does, so a single bad client degrades the round instead of
        # taking down the run. (Found and fixed on feature/cifar100-scaling after
        # two CIA retry combos crashed this way; ported to feature/eurosat-scaling
        # since EuroSAT's global-dp combos share this exact code path.)
        if not reply.has_content():
            continue
        model = reply.content.get(arrayrecord_key)
        if not isinstance(model, ArrayRecord):
            continue
        metrics = next(iter(reply.content.metric_records.values()), MetricRecord())
        client_id = int(metrics.get("client-id", fallback_id))
        rows.append((client_id, model_update_l2_norm(model, current_arrays)))
    rows.sort(key=lambda row: row[0])
    norms = [norm for _, norm in rows]
    clipped = [int(norm > clipping_norm) for norm in norms]
    return {
        "dp-client-ids": [client_id for client_id, _ in rows],
        "dp-update-norms-before-clipping": norms,
        "dp-client-clipped": clipped,
        "dp-num-clients-clipped": sum(clipped),
        "dp-fraction-clipped": sum(clipped) / len(clipped) if clipped else 0.0,
    }


def signal_to_noise_diagnostics(
    aggregated_arrays: ArrayRecord,
    *,
    current_arrays: ArrayRecord,
    noise_stdv: float,
) -> dict[str, float | int]:
    """Describe the server update magnitude relative to expected noise L2 norm."""
    signal_norm = model_update_l2_norm(aggregated_arrays, current_arrays)
    count = parameter_count(aggregated_arrays)
    expected_noise_norm = float(noise_stdv * np.sqrt(count))
    diagnostics: dict[str, float | int] = {
        "dp-signal-update-norm": signal_norm,
        "dp-parameter-count": count,
        "dp-expected-noise-l2-norm": expected_noise_norm,
    }
    if signal_norm > 0.0:
        diagnostics["dp-noise-to-signal-ratio"] = expected_noise_norm / signal_norm
        diagnostics["dp-noise-percent-of-signal"] = (
            100.0 * expected_noise_norm / signal_norm
        )
    return diagnostics


def add_diagnostics(metrics: MetricRecord | None, values: dict) -> MetricRecord:
    """Add JSON-compatible diagnostic values to a Flower metric record."""
    output = metrics if metrics is not None else MetricRecord()
    for key, value in values.items():
        output[key] = value
    return output


def leave_one_out_residual_diagnostics(
    replies: Sequence[Message],
    *,
    current_arrays: ArrayRecord,
    arrayrecord_key: str = "arrays",
) -> dict[str, list[float] | list[int]]:
    """Norm of each client's update minus the equal-weight mean of the others' updates.

    With ``u_i = model_i - current_arrays`` and ``S = sum_j u_j`` over the ``n``
    successful replies, the residual is ``u_i - (S - u_i) / (n - 1)
    = (n u_i - S) / (n - 1)``. Measured on the raw replies, i.e. before any
    clipping. Error replies and replies without an ArrayRecord are skipped, as
    in ``clipping_diagnostics``; fewer than two usable replies give empty lists.
    """
    reference = [
        np.asarray(array, dtype=np.float64) for array in current_arrays.to_numpy_ndarrays()
    ]
    rows: list[tuple[int, list[np.ndarray]]] = []
    for fallback_id, reply in enumerate(replies):
        if not reply.has_content():
            continue
        model = reply.content.get(arrayrecord_key)
        if not isinstance(model, ArrayRecord):
            continue
        arrays = model.to_numpy_ndarrays()
        if len(arrays) != len(reference) or any(
            array.shape != base.shape for array, base in zip(arrays, reference, strict=True)
        ):
            raise ValueError("Model and reference arrays must have matching shapes.")
        metrics = next(iter(reply.content.metric_records.values()), MetricRecord())
        client_id = int(metrics.get("client-id", fallback_id))
        update = [
            np.asarray(array, dtype=np.float64) - base
            for array, base in zip(arrays, reference, strict=True)
        ]
        rows.append((client_id, update))
    rows.sort(key=lambda row: row[0])
    count = len(rows)
    if count < 2:
        return {"loo-client-ids": [], "loo-residual-norms": []}
    totals = [sum(update[layer] for _, update in rows) for layer in range(len(reference))]
    norms = [
        float(np.sqrt(sum(
            float(np.sum(np.square(count * layer - total)))
            for layer, total in zip(update, totals, strict=True)
        ))) / (count - 1)
        for _, update in rows
    ]
    return {
        "loo-client-ids": [client_id for client_id, _ in rows],
        "loo-residual-norms": norms,
    }


class LeaveOneOutResidualLogging(Strategy):
    """Wrap any strategy and log ``leave_one_out_residual_diagnostics`` every round.

    The wrapper sits outside the DP wrappers, so it sees the replies before they
    are clipped, and it also works for vanilla strategies. Training and noise are
    untouched; the only change is two extra keys in the round's train metrics.
    """

    def __init__(self, strategy: Strategy, arrayrecord_key: str = "arrays") -> None:
        self.strategy = strategy
        self.arrayrecord_key = arrayrecord_key
        self.current_arrays = ArrayRecord()

    def __repr__(self) -> str:
        return f"LeaveOneOutResidualLogging({self.strategy!r})"

    def summary(self) -> None:
        log(INFO, "\t├──> Leave-one-out residual norms logged every round")
        self.strategy.summary()

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        self.current_arrays = arrays
        return self.strategy.configure_train(server_round, arrays, config, grid)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        reply_list = list(replies)
        diagnostics = leave_one_out_residual_diagnostics(
            reply_list,
            current_arrays=self.current_arrays,
            arrayrecord_key=self.arrayrecord_key,
        )
        arrays, metrics = self.strategy.aggregate_train(server_round, reply_list)
        return arrays, add_diagnostics(metrics, diagnostics)

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        return self.strategy.configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None:
        return self.strategy.aggregate_evaluate(server_round, replies)
