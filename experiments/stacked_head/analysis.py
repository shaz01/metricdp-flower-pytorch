"""Summarise finished stacked-head cells and apply the pre-stated gate (per task and risk, budget by budget)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

GATE_MIN_CELLS = 5
GATE_MIN_GAIN = 0.001
GATE_WORST_CELL = -0.003


def load_cells(directory: Path, tag: str | None = None) -> list[dict]:
    cells = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name.endswith("_attack.json") or "_attack_" in path.name:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("summary") is None or "config" not in result:
            continue
        if tag is not None and not path.stem.endswith(f"_{tag}"):
            continue
        cells.append(result)
    return cells


def gate(gains: list[float], cells_expected: int = 6) -> dict:
    """Gate used in the research phase: >= 5/6 cells above .001, mean > 0, no cell below -.003."""
    above = sum(g > GATE_MIN_GAIN for g in gains)
    return {
        "cells": len(gains), "cells_above_threshold": int(above), "mean_gain": float(np.mean(gains)), "worst_cell": float(min(gains)),
        "passed": bool(len(gains) >= cells_expected and above >= GATE_MIN_CELLS and np.mean(gains) > 0 and min(gains) >= GATE_WORST_CELL),
    }


def summarize(cells: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for result in cells:
        config = result["config"]
        groups[(config["task"], int(config["budget"]), float(config["risk"]))].append(result)
    rows = []
    for (task, budget, risk), members in sorted(groups.items()):
        gains = [m["summary"]["gain_over_control"] for m in members]
        accuracy = [m["summary"]["accuracy_delta"] for m in members]
        rows.append({"task": task, "budget": budget, "risk": risk, "mean_accuracy_delta": float(np.mean(accuracy)), **gate(gains)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("results/stacked_head"))
    parser.add_argument("--tag", default=None, help="only result files ending in _<tag>.json")
    args = parser.parse_args()
    rows = summarize(load_cells(args.directory, args.tag))
    for row in rows:
        print(f"{row['task']:<22} b{row['budget']:<4} q{row['risk']:<5} cells {row['cells']}  >.001: {row['cells_above_threshold']}/{row['cells']}  "
              f"mean {row['mean_gain']:+.4f}  worst {row['worst_cell']:+.4f}  acc {row['mean_accuracy_delta']:+.4f}  gate {'PASS' if row['passed'] else 'not met'}")


if __name__ == "__main__":
    main()
