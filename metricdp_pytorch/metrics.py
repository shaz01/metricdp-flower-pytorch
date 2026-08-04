"""Metric aggregation which preserves aligned per-client scalar histories."""

from __future__ import annotations

from flwr.app import MetricRecord, RecordDict
from flwr.serverapp.strategy.strategy_utils import aggregate_metricrecords


def aggregate_metrics_with_clients(
    records: list[RecordDict], weighting_metric_name: str
) -> MetricRecord:
    """Return weighted aggregates plus one aligned list per client metric."""
    rows: list[tuple[int, MetricRecord, RecordDict]] = []
    for record in records:
        metrics = next(iter(record.metric_records.values()))
        if "client-id" not in metrics:
            raise ValueError(
                "Metric records must include a 'client-id' for deterministic "
                "aggregation."
            )
        rows.append((int(metrics["client-id"]), metrics, record))
    rows.sort(key=lambda row: row[0])

    aggregated = aggregate_metricrecords(
        [record for _, _, record in rows], weighting_metric_name
    )
    aggregated.pop("client-id", None)
    aggregated["client-ids"] = [client_id for client_id, _, _ in rows]
    keys = sorted(
        {
            key
            for _, metrics, _ in rows
            for key, value in metrics.items()
            if key != "client-id" and not isinstance(value, list)
        }
    )
    for key in keys:
        if all(key in metrics for _, metrics, _ in rows):
            aggregated[f"per-client-{key}"] = [
                metrics[key] for _, metrics, _ in rows
            ]
    return aggregated
