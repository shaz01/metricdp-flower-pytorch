"""P3 figures from ``analysis.json`` and the follow-up analyses (no runs).

    PYTHONPATH="$PWD" uv run --no-sync --with matplotlib python results/cia_v3/robustness/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
ATTACKERS = (("standard_11", "Standard, 11 checkpoints"), ("all_51", "All 51 checkpoints"),
             ("sampling_aware_11", "Sampling-aware, 5-round windows"), ("all_51_plus_z", "51 checkpoints + schedule"),
             ("every_round", "Every-round sampling-aware (G-P3(a))"))


def _style(ax, axis="x"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def attackers(analysis):
    """Claim 1: composite minus metric privacy at 0.740 under each attacker."""
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    _style(ax)
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.axvline(-0.05, color=MUTED, lw=1, ls=(0, (6, 3)), zorder=1)
    ax.text(-0.05, -0.55, "G-C1: at most -0.05 ", color=MUTED, fontsize=8, ha="right", va="bottom")
    for y, (key, label) in enumerate(ATTACKERS):
        gate = analysis["matched"][key]["0.74"]["gate"]
        color = ORANGE if key == "every_round" else BLUE
        ax.hlines(y, gate["low"], gate["high"], color=color, lw=2, zorder=2)
        ax.plot([gate["estimate"]], [y], marker="o", ms=7, color=color, mec="white", mew=1.5, zorder=3)
    ax.set_yticks(range(len(ATTACKERS)), [label for _, label in ATTACKERS], color=MUTED, fontsize=9)
    ax.set_ylim(-0.6, len(ATTACKERS) - 0.3)
    ax.set_xlabel("Composite minus metric privacy at accuracy 0.740 (AUC, 97.5% jackknife)", color=MUTED, fontsize=9)
    ax.set_title("EuroSAT: seeing every round reverses the composite's advantage", color=INK, fontsize=10, loc="left", x=-0.45)
    fig.tight_layout()
    return _save(fig, "composite_minus_mp_by_attacker.png")


def in_count(follow_up, extended):
    """Claim 2: AUC of the IN-count configurations, at R = 32 and R = 64."""
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    _style(ax, axis="y")
    names = ("8 IN, 24 active", "24 IN, 24 active", "24 IN, 40 active")
    for offset, (label, color, configs) in zip((-0.1, 0.1), (("R = 32", BLUE, follow_up["configs"]),
                                                             ("R = 64", ORANGE, extended["configs"]))):
        for x, name in enumerate(names):
            if name not in configs:
                continue
            auc = configs[name]["all"]["auc"]
            ax.vlines(x + offset, auc["low"], auc["high"], color=color, lw=2, zorder=2)
            ax.plot([x + offset], [auc["estimate"]], marker="o", ms=7, color=color, mec="white", mew=1.5, ls="none",
                    zorder=3, label=label if x == 0 else None)
    ax.set_xticks(range(len(names)), names, color=MUTED, fontsize=9)
    ax.set_xlim(-0.5, len(names) - 0.5)
    ax.set_ylabel("Attack AUC, global DP 2x (97.5% block jackknife)", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    d = extended["d_int"]
    ax.set_title(f"EuroSAT: at a fixed active count the IN count moves AUC by {d['estimate']:+.3f} "
                 f"[{d['low']:+.3f}, {d['high']:+.3f}]", color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    return _save(fig, "in_count.png")


def _save(fig, name):
    out = HERE / "figures" / name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    load = lambda name: json.loads((HERE / name).read_text())  # noqa: E731
    for path in (attackers(load("analysis.json")),
                 in_count(load("analysis_follow_up.json"), load("analysis_follow_up_r64.json"))):
        print(path)


if __name__ == "__main__":
    main()
