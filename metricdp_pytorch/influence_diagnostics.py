"""Per-client influence diagnostics, logged next to aggregation without changing it.

``per_client_influence`` is a pure function. ``InfluenceLoggingStrategy`` wraps any
Flower strategy (vanilla or a server-side DP wrapper): it reads the client replies
*before* the wrapped strategy clips them, delegates aggregation unchanged, and
adds aligned per-client lists to the round's train metrics. It draws no random
numbers and never mutates replies, so training results are identical with or
without it (see ``tests/test_influence_diagnostics.py``).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Message, MetricRecord
from flwr.serverapp import Grid
from flwr.serverapp.strategy import Strategy

PREFIX = "influence-"
# Train-metric keys added every round; each is a list aligned with ``influence-client-ids``.
FIELDS = (
    "update_norm_before_clipping",
    "clipped",
    "distance_from_weighted_clipped_average",
    "cosine_with_weighted_clipped_average",
    "leave_one_out_influence_norm",
    "aggregation_weight",
    "num_examples",
)


def metric_key(field: str) -> str:
    return PREFIX + field.replace("_", "-")


def per_client_influence(updates, client_ids, num_examples, clipping_norm, weights=None):
    """Return ID-keyed diagnostics; ``updates`` are flattened ``model - global`` vectors.

    Clipping follows Flower's fixed clipping (scale ``min(1, C/||u||)``); with
    ``clipping_norm=None`` (vanilla) nothing is clipped. ``weights`` are the raw
    FedAvg weights (the strategy's ``weighted_by_key`` metric; default
    ``num_examples``), normalized to sum to 1. ``g`` is the weighted average of
    clipped updates, exactly as FedAvg aggregates them. Leave-one-out influence is
    ``||w/(1-w) (u - g)||``: how far ``g`` moves when this client is dropped and
    the others are reweighted.
    """
    weights = num_examples if weights is None else weights
    updates = [np.asarray(u, dtype=np.float64).reshape(-1) for u in updates]
    if not (len(updates) == len(client_ids) == len(num_examples) == len(weights)) or not updates:
        raise ValueError("Updates, IDs and example counts must be nonempty and aligned")
    if len(set(client_ids)) != len(client_ids) or any(n <= 0 for n in num_examples):
        raise ValueError("Client IDs must be unique and example counts positive")
    if any(w <= 0 for w in weights):
        raise ValueError("Aggregation weights must be positive")
    if any(u.shape != updates[0].shape for u in updates):
        raise ValueError("Updates must have equal shapes")
    if clipping_norm is not None and clipping_norm <= 0:
        raise ValueError("Clipping norm must be positive or None")
    norms = [float(np.linalg.norm(u)) for u in updates]
    clipped = [u * min(1.0, clipping_norm / norm) if clipping_norm is not None and norm else u
               for u, norm in zip(updates, norms, strict=True)]
    total = float(sum(weights))
    weights = [float(w) / total for w in weights]
    average = np.zeros_like(clipped[0])
    for w, c in zip(weights, clipped, strict=True):
        average += w * c
    average_norm = float(np.linalg.norm(average))
    rows = {}
    for cid, n, w, norm, c in zip(client_ids, num_examples, weights, norms, clipped, strict=True):
        distance = float(np.linalg.norm(c - average))
        c_norm = float(np.linalg.norm(c))
        rows[int(cid)] = {
            "update_norm_before_clipping": norm,
            "clipped": clipping_norm is not None and norm > clipping_norm,
            "distance_from_weighted_clipped_average": distance,
            "cosine_with_weighted_clipped_average": (
                float(np.dot(c, average) / (c_norm * average_norm)) if c_norm and average_norm else 0.0),
            "leave_one_out_influence_norm": w / (1.0 - w) * distance if w < 1.0 else 0.0,
            "aggregation_weight": w,
            "num_examples": int(n),
        }
    return rows


def influence_metrics(rows: dict) -> dict[str, list]:
    """Flatten ``per_client_influence`` output into MetricRecord-compatible lists."""
    ids = sorted(rows)
    out: dict[str, list] = {PREFIX + "client-ids": ids}
    for field in FIELDS:
        values = [rows[i][field] for i in ids]
        if field == "clipped":
            values = [int(v) for v in values]
        out[metric_key(field)] = values
    return out


def _flat_update(model: ArrayRecord, reference: Sequence[np.ndarray]) -> np.ndarray:
    arrays = model.to_numpy_ndarrays()
    if len(arrays) != len(reference) or any(a.shape != r.shape for a, r in zip(arrays, reference)):
        raise ValueError("Client and global arrays must have matching shapes")
    return np.concatenate([(a.astype(np.float64) - r.astype(np.float64)).reshape(-1)
                           for a, r in zip(arrays, reference, strict=True)])


def reply_influence(replies: Sequence[Message], reference: ArrayRecord,
                    clipping_norm: float | None, id_map: Sequence[int] | None = None,
                    arrayrecord_key: str = "arrays", weight_key: str = "num-examples"):
    """Diagnostics for the successful replies; IDs translated through ``id_map``.

    ``weight_key`` is the reply metric FedAvg weights by (``num-examples`` or, for
    equal weighting, ``unit-weight``); ``num_examples`` is always logged as is.
    """
    current = reference.to_numpy_ndarrays()
    updates, ids, counts, weights = [], [], [], []
    for reply in replies:
        if reply.has_error() or not reply.has_content():
            continue
        model = reply.content.get(arrayrecord_key)
        if not isinstance(model, ArrayRecord):
            continue
        metrics = next(iter(reply.content.metric_records.values()))
        cid = int(metrics["client-id"])
        ids.append(id_map[cid] if id_map is not None else cid)
        counts.append(float(metrics["num-examples"]))
        weights.append(float(metrics[weight_key]))
        updates.append(_flat_update(model, current))
    if not updates:
        return {}
    return influence_metrics(per_client_influence(updates, ids, counts, clipping_norm, weights))


def aggregation_weight_key(strategy: Strategy) -> str:
    """The ``weighted_by_key`` of the innermost strategy (DP wrappers nest via ``.strategy``)."""
    while not hasattr(strategy, "weighted_by_key") and hasattr(strategy, "strategy"):
        strategy = strategy.strategy
    return str(getattr(strategy, "weighted_by_key", "num-examples"))


class InfluenceLoggingStrategy(Strategy):
    """Delegate everything to ``strategy``; add per-client influence to train metrics."""

    def __init__(self, strategy: Strategy, *, clipping_norm: float | None,
                 id_map: Sequence[int] | None = None) -> None:
        self.strategy = strategy
        self.clipping_norm = clipping_norm
        self.id_map = tuple(id_map) if id_map is not None else None
        self.current_arrays = ArrayRecord()
        self.weight_key = aggregation_weight_key(strategy)

    def summary(self) -> None:
        self.strategy.summary()

    def configure_train(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord,
                        grid: Grid) -> Iterable[Message]:
        self.current_arrays = arrays
        return self.strategy.configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round: int, replies: Iterable[Message]):
        reply_list = list(replies)
        # Measure before delegating: DP wrappers clip the replies in place.
        diagnostics = reply_influence(reply_list, self.current_arrays, self.clipping_norm, self.id_map,
                                      weight_key=self.weight_key)
        arrays, metrics = self.strategy.aggregate_train(server_round, reply_list)
        if diagnostics:
            metrics = metrics if metrics is not None else MetricRecord()
            for key, value in diagnostics.items():
                metrics[key] = value
        return arrays, metrics

    def configure_evaluate(self, server_round: int, arrays: ArrayRecord, config: ConfigRecord,
                           grid: Grid) -> Iterable[Message]:
        return self.strategy.configure_evaluate(server_round, arrays, config, grid)

    def aggregate_evaluate(self, server_round: int, replies: Iterable[Message]):
        return self.strategy.aggregate_evaluate(server_round, replies)
