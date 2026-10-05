"""Summarize Stage B: final accuracy per (dataset, partition, mechanism, ratio), median over seeds.

Also carries the per-run DP diagnostics from analyze.summarize_run (update norm, client
distance, noise) so the accuracy gap can be read next to the noise each mechanism added.
Writes summary.json next to the runs and prints one table per cell.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path

from results.cia_frontier.dataset_vs_model.analyze import summarize_run
from results.cia_frontier.dataset_vs_model.stage_b import DEFAULT_OUTPUT

_NAME = re.compile(
    r"^stage-b-(?P<dataset>[a-z0-9]+)-(?P<partition>[a-z]+)-"
    r"(?P<arm>vanilla|(?P<privacy>global-dp|metric-privacy)-r(?P<ratio>[0-9.e-]+))__"
)
_KEYS = ("final_accuracy", "update_norm", "max_pairwise_distance", "metric_noise_stdv")


def _median(values):
    values = [v for v in values if v is not None]
    return statistics.median(values) if values else None


def summarize(output: Path, *, skip_rounds: int = 0) -> dict:
    groups: dict[str, dict[str, list[dict]]] = {}
    for path in sorted(output.rglob("*.json")):
        match = _NAME.match(path.name)
        if not match or path.name.endswith(".evaluation.json"):
            continue
        data = json.loads(path.read_text())
        if "train_metrics" not in data:
            continue
        row = summarize_run(data, skip_rounds=skip_rounds)
        row["seed"] = data.get("metadata", {}).get("seed")
        cell = f"{match['dataset']}+{match['partition']}"
        groups.setdefault(cell, {}).setdefault(match["arm"], []).append(row)
    return {cell: {arm: {"seeds": len(rows), **{k: _median(r[k] for r in rows) for k in _KEYS},
                         "runs": rows}
                   for arm, rows in arms.items()}
            for cell, arms in groups.items()}


def _arm_key(arm: str) -> tuple:
    if arm == "vanilla":
        return (0, 0.0)
    privacy, ratio = arm.rsplit("-r", 1)
    return (1 if privacy == "global-dp" else 2, float(ratio))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-rounds", type=int, default=0)
    args = parser.parse_args()
    summary = summarize(args.output, skip_rounds=args.skip_rounds)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fmt = lambda v, p=3: "-" if v is None else f"{v:.{p}f}"
    for cell, arms in summary.items():
        print(f"\n{cell}\n{'arm':<26}{'seeds':>6}{'acc':>7}{'upd norm':>10}{'max dist':>10}{'MP noise':>10}")
        for arm in sorted(arms, key=_arm_key):
            r = arms[arm]
            print(f"{arm:<26}{r['seeds']:>6}{fmt(r['final_accuracy']):>7}{fmt(r['update_norm'], 2):>10}"
                  f"{fmt(r['max_pairwise_distance'], 2):>10}{fmt(r['metric_noise_stdv'], 4):>10}")


if __name__ == "__main__":
    main()
