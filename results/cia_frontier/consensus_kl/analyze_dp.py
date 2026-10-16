"""Step 3 analysis: accuracy at matched leakage, global DP with and without consensus KL 0.3 (PROTOCOL.md step 3).

    uv run --no-sync python -m results.cia_frontier.consensus_kl.analyze_dp [root] --out analysis_step3.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from results.cia_frontier.signature_suppression.analyze import cell_statistics, load

HERE = Path(__file__).resolve().parent
MECHANISMS = ("gdp", "gdp-consensus-kl-0.3")
TOP_DECILE = (23, 7)
MARGIN = 0.02
RESAMPLES = 2000
BOOTSTRAP_SEED = 1729


def split(cell):
    mechanism, ratio = cell.rsplit("-r", 1)
    return mechanism, float(ratio)


def accuracy_at(scores, accuracies, level):
    """Piecewise-linear interpolation of accuracy as a function of score; None outside the mechanism's range."""
    order = np.argsort(scores)
    s, a = np.asarray(scores)[order], np.asarray(accuracies)[order]
    if level < s[0] or level > s[-1]:
        return None
    return float(np.interp(level, s, a))


def evaluate(data, resamples=RESAMPLES):
    cells = sorted(data)
    clusters = sorted(data[cells[0]]["cells"])
    if any(sorted(data[c]["cells"]) != clusters for c in cells):
        raise ValueError("All cells must share the target-seed clusters (paired comparison)")
    grid = {m: sorted((split(c)[1], c) for c in cells if split(c)[0] == m) for m in MECHANISMS}
    accuracy = {c: float(np.mean(data[c]["accuracy"])) for c in cells}

    def scores(chosen, key="score"):
        return {c: cell_statistics([data[c]["cells"][clusters[i]]["clean"] for i in chosen])[key] for c in cells}

    everything = list(range(len(clusters)))
    point = {c: cell_statistics([data[c]["cells"][k]["clean"] for k in clusters]) for c in cells}
    noisy = {c: cell_statistics([data[c]["cells"][k]["noisy"] for k in clusters]) for c in cells}
    top = [i for i, (_, t) in enumerate(clusters) if t in TOP_DECILE]
    top_point = {c: cell_statistics([data[c]["cells"][clusters[i]]["clean"] for i in top]) for c in cells}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [rng.integers(0, len(clusters), len(clusters)) for _ in range(resamples)]
    boot = [scores(d) for d in draws]
    boot_auc = [scores(d, "auc") for d in draws]
    interval = lambda xs: [float(x) for x in np.quantile(xs, [0.025, 0.975])] if len(xs) else None

    def matched(score_map):
        out = []
        gdp_scores = [score_map[c] for _, c in grid["gdp"]]
        ckl_scores = [score_map[c] for _, c in grid["gdp-consensus-kl-0.3"]]
        for (ratio, cell), level in zip(grid["gdp"], gdp_scores):
            a_ckl = accuracy_at(ckl_scores, [accuracy[c] for _, c in grid["gdp-consensus-kl-0.3"]], level)
            out.append(None if a_ckl is None else a_ckl - accuracy[cell])
        return out

    base = matched({c: point[c]["score"] for c in cells})
    boots = [matched(b) for b in boot]
    primary = []
    for index, ((ratio, cell), delta) in enumerate(zip(grid["gdp"], base)):
        values = [b[index] for b in boots if b[index] is not None]
        ci = interval(values)
        primary.append(dict(gdp_ratio=ratio, matched_score=point[cell]["score"], delta_accuracy=delta,
                            ci95=ci, resamples_in_range=len(values)))
    valid = [p for p in primary if p["delta_accuracy"] is not None and p["ci95"] is not None]
    if not valid:
        reading = "no overlap of the score ranges (matched-noise comparison only)"
    elif any(p["ci95"][0] > 0 and p["delta_accuracy"] >= MARGIN for p in valid):
        reading = "supports consensus KL"
    elif all(p["ci95"][1] <= 0 for p in valid):
        reading = "against consensus KL"
    else:
        reading = "inconclusive"

    by_ratio = []
    for (ratio, gdp_cell), (_, ckl_cell) in zip(grid["gdp"], grid["gdp-consensus-kl-0.3"]):
        by_ratio.append(dict(
            ratio=ratio,
            score_reduction=point[gdp_cell]["score"] - point[ckl_cell]["score"],
            score_reduction_ci95=interval([b[gdp_cell] - b[ckl_cell] for b in boot]),
            auc_reduction=point[gdp_cell]["auc"] - point[ckl_cell]["auc"],
            auc_reduction_ci95=interval([b[gdp_cell] - b[ckl_cell] for b in boot_auc]),
            accuracy_difference=accuracy[ckl_cell] - accuracy[gdp_cell],
            top_decile_score_reduction=top_point[gdp_cell]["score"] - top_point[ckl_cell]["score"]))
    report = {c: dict(mechanism=split(c)[0], ratio=split(c)[1], clean=point[c], noisy=noisy[c], top_decile=top_point[c],
                      clean_ci95={"score": interval([b[c] for b in boot]), "auc": interval([b[c] for b in boot_auc])},
                      accuracy_mean=accuracy[c], accuracy_in=data[c]["in_accuracy"],
                      accuracy_range=[min(data[c]["accuracy"]), max(data[c]["accuracy"])],
                      client_accuracy_in=[dict(min=min(a.values()), p10=float(np.quantile(list(a.values()), 0.1)),
                                               mean=float(np.mean(list(a.values())))) for a in data[c]["client_accuracy"]])
              for c in cells}
    return dict(clusters=[f"{s}-{t}" for s, t in clusters], resamples=resamples, bootstrap_seed=BOOTSTRAP_SEED,
                cells=report, matched_leakage=primary, reading=reading, matched_noise=by_ratio)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=HERE / "results" / "step3")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args()
    text = json.dumps(evaluate(load(args.root), args.resamples), indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
