"""Defense x attack coverage matrix of the reviewed corpus (research/cia_review_2026-10/paper_table.csv).

Reads the hand-assigned tags in data/taxonomy_tags.json (one entry per corpus key: defense families the
paper proposes or evaluates, attack families it addresses). Run from the repository root:

    uv run --with pandas --with matplotlib python research/cia_review_2026-10/scripts/build_taxonomy_figure.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = os.environ.get("REPO_ROOT", os.getcwd())
OUT = os.path.join(ROOT, "research", "cia_review_2026-10")

# Cells this repository has already tested empirically against client participation inference
PROJECT_CELLS = {("Metric / d-privacy", "Client participation (CIA, user, subject)"),
                 ("Client-level DP", "Client participation (CIA, user, subject)"),
                 ("Data-dependent calibration", "Client participation (CIA, user, subject)")}


def main():
    spec = json.load(open(os.path.join(OUT, "data/taxonomy_tags.json")))
    DEF, ATK, T = spec["defense"], spec["attack"], spec["tags"]
    mat = pd.DataFrame(0, index=DEF, columns=ATK)
    atk_only = pd.Series(0, index=ATK)
    for _, (ds, as_) in T.items():
        for a in as_:
            if not ds:
                atk_only[a] += 1
            for d in ds:
                mat.loc[d, a] += 1
    mat.to_csv(os.path.join(OUT, "data/defense_attack_matrix.csv"))

    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "savefig.dpi": 300, "axes.titlelocation": "left"})
    fig, (ax0, ax) = plt.subplots(2, 1, figsize=(7.2, 5.4), gridspec_kw=dict(height_ratios=[1, 7.5], hspace=0.05), sharex=True)
    vmax = max(int(mat.values.max()), int(atk_only.max()))
    ax0.imshow(atk_only.values[None, :], cmap="Greys", vmin=0, vmax=vmax * 1.6, aspect="auto")
    for j, v in enumerate(atk_only.values):
        ax0.text(j, 0, str(v), ha="center", va="center", fontsize=7)
    ax0.set_yticks([0], ["attack-only papers"])
    ax0.tick_params(axis="x", bottom=False, labelbottom=False)
    for s in ax0.spines.values():
        s.set_visible(False)
    ax0.set_title("Defenses against client-participation inference are rarely studied beyond noise (reviewed corpus)")
    im = ax.imshow(mat.values, cmap="Blues", vmin=0, vmax=vmax * 1.2, aspect="auto")
    for i in range(len(DEF)):
        for j in range(len(ATK)):
            v = mat.values[i, j]
            ax.text(j, i, str(v) if v else "0", ha="center", va="center", fontsize=7,
                    color="white" if v >= 0.6 * vmax else ("0.55" if v == 0 else "black"))
            if (DEF[i], ATK[j]) in PROJECT_CELLS:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#DD8452", lw=2))
    cia = ATK.index("Client participation (CIA, user, subject)")
    ax.add_patch(Rectangle((cia - 0.5, -0.5), 1, len(DEF), fill=False, ec="k", lw=1.0, ls="--"))
    ax.set_yticks(range(len(DEF)), DEF)
    short = {"Record MIA": "Record\nMIA", "Client participation (CIA, user, subject)": "Client\nparticipation\n(CIA, user, subject)",
             "Source inference": "Source\ninference", "Property / distribution inference": "Property /\ndistribution\ninference",
             "Reconstruction": "Recon-\nstruction", "Auditing / lower bounds": "Auditing /\nlower bounds"}
    ax.set_xticks(range(len(ATK)), [short.get(a, a) for a in ATK], rotation=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(1.0, -0.17, "orange box = already tested in this repository; dashed column = this project's threat",
            transform=ax.transAxes, ha="right", va="top", fontsize=6)
    cb = fig.colorbar(im, ax=[ax0, ax], fraction=0.025, pad=0.02)
    cb.set_label("papers in corpus addressing the pair", fontsize=7)
    fig.savefig(os.path.join(OUT, "figures/defense_attack_coverage.png"), bbox_inches="tight")
    plt.close(fig)
    return mat, atk_only


if __name__ == "__main__":
    m, a = main()
    print(m.to_string())
