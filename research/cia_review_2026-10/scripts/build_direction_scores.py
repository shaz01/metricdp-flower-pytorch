"""Score chart for the candidate directions in research_directions.md.

Author scores are the judgement written in research_directions.md; reviewer scores come from the
adversarial reviewer pass stored in data/direction_reviewer_pass.json (a reasoning-model reviewer prompted
with the project evidence; see the document for its objections). Run from the repository root:

    uv run --with pandas --with matplotlib python research/cia_review_2026-10/scripts/build_direction_scores.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("REPO_ROOT", os.getcwd())
OUT = os.path.join(ROOT, "research", "cia_review_2026-10")

CRIT = ["novelty", "expected_gain_vs_metric_privacy", "feasibility", "theoretical_soundness"]
CRIT_LABEL = ["novelty", "expected gain\nvs metric privacy", "feasibility", "theoretical\nsoundness"]
NAMES = {"D1": "D1 signature-suppressing training\nunder fixed client-level DP",
         "D2": "D2 defense-aware CIA audit\n(side channel, increment attacker)",
         "D3": "D3 client-level f-MIP theory",
         "D4": "D4 robust private dispersion\n+ residual clipping",
         "D5": "D5 leakage-profile temporal\nallocation",
         "D6": "D6 low-dimensional sharing"}
AUTHOR = {"D1": [3, 4, 4, 4], "D2": [4, 3, 3, 3], "D3": [3, 2, 3, 3], "D4": [3, 3, 3, 3], "D5": [2, 2, 4, 3], "D6": [2, 2, 3, 3]}


def main():
    rev = json.load(open(os.path.join(OUT, "data/direction_reviewer_pass.json")))
    rows = []
    for k in NAMES:
        for c, a in zip(CRIT, AUTHOR[k]):
            rows.append(dict(direction=k, criterion=c, author=a, reviewer=rev[k]["scores"][c]))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "data/direction_scores.csv"), index=False)
    tot = df.groupby("direction")[["author", "reviewer"]].mean()
    tot["combined"] = tot.mean(axis=1)
    order = tot.sort_values("combined", ascending=False).index.tolist()

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                         "savefig.dpi": 300, "axes.titlelocation": "left"})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.4), gridspec_kw=dict(width_ratios=[3.2, 1], wspace=0.08), sharey=True)
    for yi, k in enumerate(order):
        g = df[df.direction == k]
        for xi, c in enumerate(CRIT):
            r = g[g.criterion == c].iloc[0]
            ax.plot([xi, xi], [yi - 0.12, yi + 0.12], color="0.85", lw=6, solid_capstyle="round", zorder=1)
            ax.scatter(xi - 0.12, yi, s=10 + 14 * r.author, color="#4C72B0", zorder=3)
            ax.scatter(xi + 0.12, yi, s=10 + 14 * r.reviewer, color="#DD8452", marker="s", zorder=3)
            ax.text(xi - 0.12, yi + 0.3, str(r.author), ha="center", fontsize=6, color="#4C72B0")
            ax.text(xi + 0.12, yi + 0.3, str(r.reviewer), ha="center", fontsize=6, color="#DD8452")
    ax.set_xticks(range(len(CRIT)), CRIT_LABEL)
    ax.set_yticks(range(len(order)), [NAMES[k] for k in order])
    ax.invert_yaxis()
    ax.set_xlim(-0.5, len(CRIT) - 0.5)
    ax.scatter([], [], color="#4C72B0", s=30, label="author score (1-5)")
    ax.scatter([], [], color="#DD8452", marker="s", s=30, label="adversarial reviewer (1-5)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=6.5)
    ax.set_title("D1 ranks first under both scorers; the reviewer is mostly lower on novelty and soundness")
    y = np.arange(len(order))
    ax2.barh(y, tot.loc[order, "combined"], color="0.6", height=0.5)
    for yi, k in enumerate(order):
        ax2.text(tot.loc[k, "combined"] + 0.05, yi, f"{tot.loc[k, 'combined']:.2f}", va="center", fontsize=6.5)
    ax2.set_xlim(0, 5)
    ax2.set_xlabel("mean of both scorers")
    ax2.tick_params(axis="y", left=False)
    fig.savefig(os.path.join(OUT, "figures/direction_scores.png"), bbox_inches="tight")
    plt.close(fig)
    return df, tot


if __name__ == "__main__":
    d, t = main()
    print(t.round(2).to_string())
