"""P2 figures from ``analysis.json`` (no runs).

    PYTHONPATH="$PWD" uv run --no-sync --with matplotlib python results/cia_v3/cifar10/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SERIES = {"gdp": ("Global DP", "#2a78d6"), "mp": ("Metric privacy", "#eb6834"), "comp": ("Composite", "#1baf7a")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
THRESHOLD = -0.05  # G-C1


def _style(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def curves(analysis):
    """Claim 1: AUC against accuracy per mechanism, with the two targets."""
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _style(ax)
    for target, name in zip(analysis["targets"], ("a2", "a4")):
        ax.axvline(target, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
        ax.text(target, 0.655, f" {name} = {target:.3f}", color=MUTED, fontsize=8, va="bottom")
    for prefix, (label, color) in SERIES.items():
        levels = sorted((s["accuracy"]["estimate"], s["all"]["auc"]) for name, s in analysis["levels"].items()
                        if name.startswith(prefix + "-"))
        x = [a for a, _ in levels]
        ax.plot(x, [auc["estimate"] for _, auc in levels], color=color, lw=2, marker="o", ms=6, mec="white", mew=1.5,
                zorder=3, label=label)
        ax.vlines(x, [auc["low"] for _, auc in levels], [min(auc["high"], 1) for _, auc in levels], color=color, lw=1,
                  alpha=0.6, zorder=2)
    vanilla = analysis["levels"]["vanilla"]
    ax.plot([vanilla["accuracy"]["estimate"]], [vanilla["all"]["auc"]["estimate"]], marker="*", ms=12, color=MUTED,
            mec="white", ls="none", zorder=4, label="Vanilla (no noise)")
    ax.set_ylim(0.65, 1.01)
    ax.set_xlabel("Mean final server-test accuracy (32 replicates)", color=MUTED)
    ax.set_ylabel("Attack AUC, all candidates (97.5% jackknife)", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("CIFAR-10: metric privacy's curve lies on global DP's; the composite's sits just below",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    return _save(fig, "auc_vs_accuracy.png")


def differences(analysis):
    """Claim 2: composite minus metric privacy at the targets, per attacker, against G-C1's threshold."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    _style(ax)
    ax.axhline(0, color=MUTED, lw=1, zorder=1)
    ax.axhline(THRESHOLD, color=MUTED, lw=1, ls=(0, (6, 3)), zorder=1)
    ax.text(-0.48, THRESHOLD, " G-C1: at most -0.05", color=MUTED, fontsize=8, va="bottom", ha="left")
    attackers = (("Standard (gated)", "#1baf7a",
                  lambda t: analysis["g_c1_standard"]["all"][t]["difference"]),
                 ("Sampling-aware (gated)", "#2a78d6", lambda t: analysis["g_c1_sampling_aware"][t]["gate"]),
                 ("All 51 checkpoints (reported)", MUTED,
                  lambda t: analysis["reported_composite_all_checkpoints"][t]["gate"]))
    for i, target in enumerate(map(str, analysis["targets"])):
        for offset, (label, color, get) in zip((-0.15, 0, 0.15), attackers):
            gate = get(target)
            ax.vlines(i + offset, gate["low"], gate["high"], color=color, lw=2, zorder=2)
            ax.plot([i + offset], [gate["estimate"]], marker="o", ms=7, color=color, mec="white", mew=1.5, ls="none",
                    zorder=3, label=label if i == 0 else None)
    ax.set_xticks([0, 1], [f"a2 = {analysis['targets'][0]:.3f}", f"a4 = {analysis['targets'][1]:.3f}"], color=MUTED)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("Composite minus metric privacy, AUC (97.5%)", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("CIFAR-10: the composite's reduction reaches 0.05 only once",
                 color=INK, fontsize=10, loc="left")
    fig.tight_layout()
    return _save(fig, "composite_minus_mp.png")


def _save(fig, name):
    out = HERE / "figures" / name
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main():
    analysis = json.loads((HERE / "analysis.json").read_text())
    for path in (curves(analysis), differences(analysis)):
        print(path)


if __name__ == "__main__":
    main()
