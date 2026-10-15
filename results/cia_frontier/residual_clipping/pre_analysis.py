"""WP6 pre-analysis from committed logs only (no training): how large is a client's residual around the centre?

Residual clipping releases c_t + (1/n) sum_i clip(u_i - c_t, rho) + noise. It lowers sensitivity only if residuals are
much smaller than updates. With c_t the previous released update, the residual contains (i) the client's spread around
the round's mean update, (ii) the change of the mean update between rounds and (iii) the previous round's noise.
Committed logs give (i) exactly and (iii) exactly; (ii) is not logged and is left out, so the predicted residual is a
lower bound.

  spread_i^2 = ||u_i - u_bar||^2, from sum_i w_i ||u_i - u_bar||^2 = sum_i w_i ||u_i||^2 - ||u_bar||^2 (exact for the
               weighted mean of the clipped updates; per-client spreads need vectors, so the weighted mean is used)
  noise^2    = logged expected noise L2 norm of the previous round (sqrt(d) sigma)

    uv run --no-sync python -m results.cia_frontier.residual_clipping.pre_analysis
Writes pre_analysis.csv beside this file.
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
C = 5.0


def rows_for(label, path):
    metrics = json.load(open(path))["train_metrics"]
    rounds = sorted(metrics, key=int)
    out = []
    for previous, current in zip(rounds, rounds[1:]):
        m = metrics[current]
        norms = np.minimum(np.asarray(m["dp-update-norms-before-clipping"], float), C)
        weights = np.asarray(m.get("per-client-unit-weight") or m["per-client-num-examples"], float)
        weights = weights / weights.sum()
        mean_norm = float(m["dp-signal-update-norm"])
        spread2 = max(float(weights @ norms ** 2) - mean_norm ** 2, 0.0)
        noise = float(metrics[previous]["dp-expected-noise-l2-norm"])
        out.append(dict(update_norm=float(np.median(norms)), mean_update_norm=mean_norm,
                        spread=np.sqrt(spread2), noise_prev=noise,
                        residual_lower_bound=np.sqrt(spread2 + noise ** 2)))
    keys = out[0].keys()
    summary = {k: float(np.median([r[k] for r in out])) for k in keys}
    return dict(run=label, rounds=len(out) + 1, **{f"median_{k}": round(v, 4) for k, v in summary.items()},
                residual_over_update=round(summary["residual_lower_bound"] / summary["update_norm"], 3),
                residual_over_clip=round(summary["residual_lower_bound"] / C, 3))


def main():
    runs = []
    frontier = os.path.join(ROOT, "results/cia_frontier/eurosat_frontier/results/frontier")
    for seed in (42, 43, 44):
        for path in glob.glob(f"{frontier}/global-dp-r0.001546-seed-{seed}-in/*/*.json"):
            if path.endswith("eurosat_cnn.json"):
                runs.append(rows_for(f"frontier 48 clients global DP seed {seed} (size weights)", path))
    audit = os.path.join(ROOT, "results/cia_frontier/side_channel_audit/results/eurosat")
    for clients in (3, 5, 8):
        for path in sorted(glob.glob(f"{audit}/clients-{clients}/global-dp/*-in-*/*.json")):
            if path.endswith("eurosat_cnn.json"):
                seed = path.split("__seed-")[1].split("__")[0]
                runs.append(rows_for(f"WP1 {clients} clients global DP seed {seed} (equal weights)", path))
    path = os.path.join(HERE, "pre_analysis.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        writer.writerows(runs)
    print(path)
    for row in runs:
        print(row)


if __name__ == "__main__":
    main()
