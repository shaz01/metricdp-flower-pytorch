"""Metric aggregation which preserves aligned per-client scalar histories."""

from __future__ import annotations

from flwr.app import MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords


def aggregate_metrics_with_clients(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    """Return weighted aggregates plus one aligned list per client metric."""
    aggregated = aggregate_metricrecords(records, weighting_metric_name)
    aggregated.pop("client-id", None)

    rows: list[tuple[int, MetricRecord]] = []
    for fallback_id, record in enumerate(records):
        metrics = next(iter(record.metric_records.values()))
        rows.append((int(metrics.get("client-id", fallback_id)), metrics))
    rows.sort(key=lambda row: row[0])

    aggregated["client-ids"] = [client_id for client_id, _ in rows]
    keys = sorted(
        {
            key
            for _, metrics in rows
            for key, value in metrics.items()
            if key != "client-id" and not isinstance(value, list)
        }
    )
    for key in keys:
        if all(key in metrics for _, metrics in rows):
            aggregated[f"per-client-{key}"] = [
                metrics[key] for _, metrics in rows
            ]
    return aggregated
