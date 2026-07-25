"""Print a comparison table for a completed (or in-progress) noise-multiplier sweep.

Reads whatever result JSONs already exist under ``results/noise_sweep`` and,
for each (partition, noise_multiplier), prints global-dp vs. metric-privacy
final accuracy/loss side by side plus the metric-privacy calibration distance
range, so it's easy to see whether -- and where -- the two mechanisms
diverge. Safe to run against a sweep that is still running: missing
combinations are shown as "--".
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from experiments.reproduce.sweep_noise_multiplier import (
    AGGREGATION_METHODS_SWEPT,
    NOISE_MULTIPLIERS,
    OUTPUT_DIR,
    PARTITION_MODES,
    result_path,
)


def _final_metrics(path: Path) -> dict | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    history = data.get("server_evaluate_metrics", {})
    completed = [int(k) for k in history if int(k) > 0]
    if not completed:
        return None
    metrics = dict(history[str(max(completed))])
    train_metrics = data.get("train_metrics", {})
    distances = [
        v["metric-dp-distance"]
        for k, v in train_metrics.items()
        if int(k) > 0 and "metric-dp-distance" in v
    ]
    if distances:
        metrics["_distance_mean"] = statistics.mean(distances)
        metrics["_distance_min"] = min(distances)
        metrics["_distance_max"] = max(distances)
    metrics["_rounds_completed"] = len(completed)
    return metrics


def main() -> None:
    header = (
        f"{'partition':<12} {'nm':>6} {'aggr':<8} "
        f"{'global-dp acc':>14} {'metric-dp acc':>14} {'metric-dp advantage':>20} "
        f"{'distance (min/mean/max)':>26}"
    )
    print(header)
    print("-" * len(header))
    for partition in PARTITION_MODES:
        for noise_multiplier in NOISE_MULTIPLIERS:
            for aggregation in AGGREGATION_METHODS_SWEPT:
                global_metrics = _final_metrics(
                    result_path(partition, "global-dp", aggregation, noise_multiplier)
                )
                metric_metrics = _final_metrics(
                    result_path(partition, "metric-privacy", aggregation, noise_multiplier)
                )
                global_acc = global_metrics["accuracy"] if global_metrics else None
                metric_acc = metric_metrics["accuracy"] if metric_metrics else None
                advantage = (
                    f"{metric_acc - global_acc:+.4f}"
                    if global_acc is not None and metric_acc is not None
                    else "--"
                )
                distance_str = "--"
                if metric_metrics and "_distance_mean" in metric_metrics:
                    distance_str = (
                        f"{metric_metrics['_distance_min']:.3f}/"
                        f"{metric_metrics['_distance_mean']:.3f}/"
                        f"{metric_metrics['_distance_max']:.3f}"
                    )
                print(
                    f"{partition:<12} {noise_multiplier:>6} {aggregation:<8} "
                    f"{global_acc if global_acc is not None else '--':>14} "
                    f"{metric_acc if metric_acc is not None else '--':>14} "
                    f"{advantage:>20} "
                    f"{distance_str:>26}"
                )


if __name__ == "__main__":
    main()
