"""Plateau check that fixes Stage C's round count R (rule set before the runs).

Per dataset (vanilla, Dirichlet a=0.3, seed 42, 70 rounds): plateau = mean server accuracy over
rounds 60-70; R_d = earliest round whose trailing 3-round moving average of accuracy is within
0.02 of that plateau. R = max over datasets, rounded up to a multiple of 5, capped at 50; if no
dataset reaches it by round 70, R = 50.

    python -m results.cia_frontier.dataset_vs_model.plateau [--root DIR]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path("results/cia_frontier/dataset_vs_model/results/plateau")
WINDOW = (60, 70)
TOLERANCE = 0.02
CAP = 50


def plateau_round(curve: dict[int, float]) -> tuple[float, int | None]:
    plateau = sum(curve[r] for r in range(WINDOW[0], WINDOW[1] + 1)) / (WINDOW[1] - WINDOW[0] + 1)
    for r in range(3, max(curve) + 1):
        if abs(sum(curve[r - k] for k in range(3)) / 3 - plateau) <= TOLERANCE + 1e-9:
            return plateau, r
    return plateau, None


def choose_r(rounds: list[int | None]) -> tuple[int, str]:
    reached = [r for r in rounds if r is not None]
    if not reached:
        return CAP, "no dataset reached its plateau by round 70; R = cap"
    if len(reached) < len(rounds):
        return CAP, "a dataset never reached its plateau by round 70; R = cap"
    value = min(CAP, 5 * math.ceil(max(reached) / 5))
    return value, "capped at 50" if 5 * math.ceil(max(reached) / 5) > CAP else "max R_d rounded up to a multiple of 5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    datasets = {}
    for path in sorted(args.root.rglob("stage-b-*-vanilla-in__*.json")):
        if path.name.endswith(".evaluation.json"):
            continue
        dataset = path.name.split("-")[2]
        metrics = json.loads(path.read_text())["server_evaluate_metrics"]
        curve = {int(r): float(m["accuracy"]) for r, m in metrics.items() if int(r) >= 1}
        plateau, reached = plateau_round(curve)
        datasets[dataset] = {"run": path.name, "accuracy": [round(curve[r], 4) for r in sorted(curve)],
                             "plateau_mean_60_70": round(plateau, 4), "R_d": reached,
                             "accuracy_at": {r: round(curve[r], 4) for r in (10, 20, 30, 40, 50, 60, 70)}}
    if sorted(datasets) != ["cifar10s", "eurosat32"]:
        raise SystemExit(f"Expected cifar10s and eurosat32 runs, found {sorted(datasets)}")
    value, note = choose_r([d["R_d"] for d in datasets.values()])
    result = {"rule": __doc__.split("\n\n")[1].replace("\n", " "), "datasets": datasets, "R": value, "note": note}
    (args.root / "plateau.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("plateau_mean_60_70", "R_d", "accuracy_at")}
                      for k, v in datasets.items()} | {"R": value, "note": note}))


if __name__ == "__main__":
    main()
