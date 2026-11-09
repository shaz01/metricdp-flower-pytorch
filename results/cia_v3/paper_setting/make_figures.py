"""P1 figures from ``analysis.json`` (no runs).

    PYTHONPATH="$PWD" uv run --no-sync --with matplotlib python results/cia_v3/paper_setting/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SERIES = {"gdp": ("Global DP", "#2a78d6"), "mp": ("Metric privacy", "#eb6834")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PAPER_TABLE_13 = {"gdp": 0.397, "mp": 0.493}  # Sáinz-Pardo Díaz et al. 2026, Table 13 (one IN/OUT trajectory each)
COLUMNS = (("Paper,\nTable 13", None), ("Ours, z = 0.01\n(paper's noise)", "z0.01"),
           ("Ours, z = 0.024\n(paper's accuracy)", "z0.024"))


def _style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.axhline(0.5, color=MUTED, lw=1, zorder=1)


def calibrated_figure(analysis):
    """Claim 1: the calibrated attacker against the paper's reported AUC, per mechanism."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _style(ax)
    for offset, (key, (label, color)) in zip((-0.12, 0.12), SERIES.items()):
        for i, (_, suffix) in enumerate(COLUMNS):
            x = i + offset
            if suffix is None:
                ax.plot([x], [PAPER_TABLE_13[key]], marker="o", ms=8, mfc="white", mec=color, mew=2, ls="none",
                        zorder=3)
                continue
            auc = analysis["configs"][f"{key}-{suffix}"]["calibrated"]
            ax.vlines(x, auc["low"], min(auc["high"], 1.0), color=color, lw=2, zorder=2)
            ax.plot([x], [auc["estimate"]], marker="o", ms=8, color=color, mec="white", mew=1.5, ls="none", zorder=3,
                    label=label if i == 1 else None)
    vanilla = analysis["configs"]["vanilla"]["calibrated"]
    ax.text(len(COLUMNS) - 0.5, 0.5, " chance", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(range(len(COLUMNS)), [c for c, _ in COLUMNS], color=MUTED, fontsize=9)
    ax.set_xlim(-0.5, len(COLUMNS) - 0.5)
    ax.set_ylim(0.3, 1.04)
    ax.set_ylabel("Attack AUC on the target client", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower right",
              title="Filled: calibrated attacker, 20 pairs (95% jackknife)\nHollow: the paper's single-draw AUC",
              title_fontsize=8, alignment="left")
    ax.set_title(f"The target client is identified under both DP mechanisms (vanilla: {vanilla['estimate']:.2f} "
                 "on the noisy shadow)", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    return _save(fig, "calibrated_vs_paper.png")


def per_pair_figure(analysis):
    """Claim 2: the paper's statistic over the 20 independent pairs, with the calibrated AUC of the same runs."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _style(ax)
    columns = COLUMNS[1:]
    for offset, (key, (label, color)) in zip((-0.15, 0.15), SERIES.items()):
        for i, (_, suffix) in enumerate(columns):
            summary = analysis["configs"][f"{key}-{suffix}"]
            values = summary["paper_statistic"]["per_pair"]
            spread = np.random.default_rng(0).uniform(-0.06, 0.06, len(values))  # horizontal jitter only
            ax.plot([i + offset + s for s in spread], values, marker="o", ms=4.5, color=color, mec="white", mew=0.6,
                    ls="none", zorder=3, label=label if i == 0 else None)
            ax.plot([i + offset - 0.09, i + offset + 0.09], [summary["calibrated"]["estimate"]] * 2, color=INK, lw=2,
                    zorder=4, label="Calibrated attacker, same runs" if (i, key) == (0, "gdp") else None)
            ax.plot([i + offset], [PAPER_TABLE_13[key]], marker="D", ms=7, mfc="white", mec=INK, mew=1.5, ls="none",
                    zorder=5, label="Paper, Table 13" if (i, key) == (0, "gdp") else None)
    ax.set_xticks(range(len(columns)), [c for c, _ in columns], color=MUTED, fontsize=9)
    ax.set_xlim(-0.5, len(columns) - 0.5)
    ax.set_ylim(0.3, 1.04)
    ax.set_ylabel("The paper's pooled-checkpoint AUC, per pair", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    ax.set_title("The paper's statistic over 20 independent replays: at the paper's accuracy it straddles 0.5",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    return _save(fig, "paper_statistic_per_pair.png")


def _save(fig, name):
    out = HERE / "figures" / name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    analysis = json.loads((HERE / "analysis.json").read_text())
    for path in (calibrated_figure(analysis), per_pair_figure(analysis)):
        print(path)


if __name__ == "__main__":
    main()
