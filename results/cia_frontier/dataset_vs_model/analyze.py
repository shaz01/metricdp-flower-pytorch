"""Summarize Stage A: per cell, how far apart client updates land and how big they are.

Reads the run JSONs the training runner writes and prints one row per cell
(median over seeds of the per-run medians over rounds). Also writes summary.json.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from results.cia_frontier.dataset_vs_model.runner import DEFAULT_OUTPUT

_NAME = re.compile(r"^stage-a-(?P<dataset>[a-z0-9]+)-(?P<model>[a-z0-9_]+?)(?:-cap\d+)?-r(?P<ratio>[0-9.e-]+)__")


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize_run(data: dict, *, skip_rounds: int = 0) -> dict:
    """Per-run medians over rounds after ``skip_rounds`` (distances settle in ~5 rounds)."""
    train = data.get("train_metrics", {})
    rounds = sorted(int(r) for r in train if int(r) > skip_rounds)
    norms, max_dist, med_dist, noise = [], [], [], []
    for r in rounds:
        m = train[str(r)]
        if m.get("dp-update-norms-before-clipping"):
            norms.append(statistics.median(m["dp-update-norms-before-clipping"]))
        if m.get("metric-dp-pairwise-distances"):
            max_dist.append(max(m["metric-dp-pairwise-distances"]))
            med_dist.append(statistics.median(m["metric-dp-pairwise-distances"]))
        if m.get("metric-dp-noise-stdv") is not None:
            noise.append(m["metric-dp-noise-stdv"])
    evals = data.get("server_evaluate_metrics", {})
    last = max((int(r) for r in evals), default=None)
    return {
        "rounds": len(rounds),
        "update_norm": _median(norms),
        "max_pairwise_distance": _median(max_dist),
        "median_pairwise_distance": _median(med_dist),
        "metric_noise_stdv": _median(noise),
        "final_accuracy": evals[str(last)].get("accuracy") if last is not None else None,
    }


def summarize(output: Path, *, skip_rounds: int = 0) -> dict:
    cells: dict[str, list[dict]] = {}
    for path in sorted(output.glob("*.json")):
        match = _NAME.match(path.name)
        if not match or path.name.endswith(".evaluation.json"):
            continue
        data = json.loads(path.read_text())
        if "train_metrics" not in data:
            continue
        row = summarize_run(data, skip_rounds=skip_rounds)
        row["seed"] = data.get("metadata", {}).get("seed")
        cells.setdefault(f"{match['dataset']}+{match['model']}", []).append(row)
    keys = ("update_norm", "max_pairwise_distance", "median_pairwise_distance",
            "metric_noise_stdv", "final_accuracy")
    return {cell: {"seeds": len(rows), **{k: _median(r[k] for r in rows) for k in keys},
                   "runs": rows} for cell, rows in cells.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-rounds", type=int, default=0,
                        help="Ignore the first N rounds when taking medians")
    args = parser.parse_args()
    summary = summarize(args.output, skip_rounds=args.skip_rounds)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{'cell':<24}{'seeds':>6}{'upd norm':>10}{'max dist':>10}{'med dist':>10}"
          f"{'MP noise':>10}{'acc':>7}")
    for cell, row in summary.items():
        fmt = lambda v, p=2: "-" if v is None else f"{v:.{p}f}"
        print(f"{cell:<24}{row['seeds']:>6}{fmt(row['update_norm']):>10}"
              f"{fmt(row['max_pairwise_distance']):>10}{fmt(row['median_pairwise_distance']):>10}"
              f"{fmt(row['metric_noise_stdv'], 4):>10}{fmt(row['final_accuracy'], 3):>7}")


if __name__ == "__main__":
    main()
