"""Exploratory look (not a gate): leakage at matched accuracy along training, per-target gaps, per-client accuracy.

Rounds are binned by the IN trajectory's server accuracy at that round; within a bin the target's clean shadow-loss
gap g = L_OUT - L_IN is averaged over targets, seeds and rounds (rounds of one run are not independent samples).

    uv run --no-sync --with matplotlib python -m results.cia_frontier.consensus_kl.explore [--seeds 42 43 44 45]
Writes explore_matched_accuracy.csv, explore_fairness.csv and figures/matched_accuracy.png beside this file.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCES = [ROOT / "results/cia_frontier/signature_suppression/results/stage_a", HERE / "results/seeds43-45"]
CONFIGS = ("vanilla", "consensus-kl-0.3", "weight-decay-5e-4", "local-epochs-1", "fixed-steps")
BINS = ((0.60, 0.70), (0.70, 0.75), (0.75, 0.78), (0.78, 0.82))


def trajectories(config, seeds):
    out = {}
    for source in SOURCES:
        for manifest_path in glob.glob(f"{source}/{config}/*/manifest.json"):
            manifest = json.load(open(manifest_path))
            if manifest["seed"] not in seeds or not os.path.exists(manifest_path.replace("manifest.json", "complete.json")):
                continue
            folder = os.path.dirname(manifest_path)
            out[(manifest["seed"], manifest["out_target"])] = dict(
                manifest=manifest, result=json.load(open(f"{folder}/{manifest['run_name']}.json")),
                rows=json.load(open(f"{folder}/measurements.json")))
    return out


def per_seed(config, seeds):
    runs = trajectories(config, seeds)
    for seed in sorted({s for s, _ in runs}):
        inside = runs.get((seed, None))
        if inside is None:
            continue
        rounds = inside["manifest"]["rounds"]
        accuracy = np.array([inside["result"]["server_evaluate_metrics"][str(r)]["accuracy"] for r in range(1, rounds + 1)])
        gaps = {}
        for target in inside["manifest"]["targets"]:
            if (seed, target) not in runs:
                continue
            loss = lambda run: {x["round"]: x["clean_loss"] for x in run["rows"] if x["target"] == target}
            li, lo = loss(inside), loss(runs[(seed, target)])
            gaps[target] = np.array([lo[r] - li[r] for r in range(1, rounds + 1)])
        clients = inside["result"]["client_evaluate_metrics"][str(rounds)]
        yield seed, accuracy, gaps, dict(zip(clients["client-ids"], clients["per-client-eval_acc"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45])
    args = parser.parse_args()
    bin_rows, fair_rows, curves = [], [], {}
    for config in CONFIGS:
        pooled = {b: [] for b in BINS}
        accs, gap_means, fair = [], [], []
        for seed, accuracy, gaps, clients in per_seed(config, set(args.seeds)):
            G = np.array(list(gaps.values()))
            for lo, hi in BINS:
                idx = np.where((accuracy >= lo) & (accuracy < hi))[0]
                if len(idx):
                    pooled[(lo, hi)].append(G[:, idx].ravel())
            accs.append(accuracy)
            gap_means.append(G.mean(axis=0))
            values = np.sort(list(clients.values()))
            fair_rows.append(dict(config=config, seed=seed, lowest=round(float(values[0]), 4),
                                  p10=round(float(np.quantile(values, 0.1)), 4), median=round(float(np.median(values)), 4),
                                  mean=round(float(values.mean()), 4),
                                  **{f"target_{t}": round(float(clients[t]), 4) for t in (23, 7, 3, 0, 1) if t in clients}))
        for (lo, hi), chunks in pooled.items():
            values = np.concatenate(chunks) if chunks else np.array([])
            bin_rows.append(dict(config=config, accuracy_bin=f"[{lo:.2f},{hi:.2f})", cells=len(values),
                                 mean_gap=round(float(values.mean()), 4) if len(values) else "",
                                 fraction_positive=round(float((values > 0).mean()), 3) if len(values) else ""))
        if accs:
            curves[config] = (np.mean(accs, axis=0), np.mean(gap_means, axis=0), len(accs))
    for name, rows in (("explore_matched_accuracy.csv", bin_rows), ("explore_fairness.csv", fair_rows)):
        with open(HERE / name, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    colors = {"vanilla": "#222", "consensus-kl-0.3": "#c0392b", "weight-decay-5e-4": "#2e86c1",
              "local-epochs-1": "#7f8c8d", "fixed-steps": "#8e44ad"}
    for config, (accuracy, gap, n) in curves.items():
        ax.plot(accuracy, gap, color=colors[config], lw=1.4, label=f"{config} ({n} seeds)")
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xlabel("server accuracy of the IN model at the round (mean over seeds)")
    ax.set_ylabel("mean clean shadow-loss gap OUT - IN (leakage)")
    ax.set_title("Leakage along training against accuracy (rounds 1-50)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    (HERE / "figures").mkdir(exist_ok=True)
    fig.savefig(HERE / "figures" / "matched_accuracy.png", dpi=150)
    for row in bin_rows:
        print(row)


if __name__ == "__main__":
    main()
