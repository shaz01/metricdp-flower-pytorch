"""P5 figure and tables from ``analysis.json`` (no runs).

    PYTHONPATH="$PWD" uv run --no-sync --with matplotlib python results/cia_v3/formal_cia/make_figures.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
SERIES = {"gdp": ("Global DP", "#2a78d6"), "mp": ("Metric privacy", "#eb6834"), "ss": ("Subsampling q = 0.25", "#1baf7a")}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def points(analysis, n, mechanism, attacker="calibrated"):
    # Vanilla is the noise-free end of global DP and metric privacy, not of subsampling (FINDINGS.md, deviation 1).
    keep = (mechanism,) if mechanism in ("ss", "vanilla") else (mechanism, "vanilla")
    rows = [r for r in analysis["configs"] if r["clients"] == n and r["mechanism"] in keep]
    out = []
    for r in rows:
        block = r["sampling_aware"] if attacker == "sampling-aware" and "sampling_aware" in r else r["calibrated"]
        auc = block["all_auc"]
        out.append((r["accuracy_mean"], auc["estimate"], *auc["jackknife_975"], r["config"]))
    return sorted(out)


def figure(analysis):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, n in zip(axes, (48, 480)):
        for target in analysis["targets"][str(n)]:
            ax.axvline(target, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
        ax.axhline(0.5, color=MUTED, lw=1, zorder=1)
        ax.axhline(0.60, color=MUTED, lw=1, ls=(0, (6, 3)), zorder=1)
        for key, (label, color) in SERIES.items():
            pts = points(analysis, n, key)
            x, y, lo, hi = ([p[i] for p in pts] for i in range(4))
            ax.plot(x, y, color=color, lw=2, marker="o", ms=6, mec="white", mew=1.5, zorder=3, label=label)
            ax.vlines(x, lo, hi, color=color, lw=1, alpha=0.6, zorder=2)
            if key == "ss":
                sa = points(analysis, n, key, "sampling-aware")
                ax.plot([p[0] for p in sa], [p[1] for p in sa], color=color, lw=2, ls=(0, (4, 2)), zorder=3,
                        label="Subsampling, sampling-aware attacker")
        ax.text(ax.get_xlim()[0], 0.602, " G-P5 threshold 0.60", color=MUTED, fontsize=8, va="bottom")
        van = points(analysis, n, "vanilla")[0]
        ax.plot([van[0]], [van[1]], marker="*", ms=12, color=MUTED, mec="white", ls="none", zorder=4, label="Vanilla (no noise)")
        ax.set_title(f"{n} clients{' (pseudo-clients)' if n == 480 else ''}", color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Mean final server-test accuracy", color=MUTED)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors=MUTED)
    axes[0].set_ylabel("Attack AUC, all candidates (97.5% jackknife)", color=MUTED)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Client inference on the frozen-feature head, 20 public images, T = 10", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    out = HERE / "figures" / "auc_vs_accuracy.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    return out


def tables(analysis):
    folder = HERE / "tables"
    folder.mkdir(exist_ok=True)
    with (folder / "configs.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clients", "config", "mu", "epsilon", "auc_bound", "accuracy", "auc", "auc_low", "auc_high",
                    "tpr_5pct", "tpr_1pct", "top5_auc", "sa_auc", "min_client", "p10_client", "implied_mu"])
        for r in analysis["configs"]:
            p, c = r["privacy"], r["calibrated"]
            sa = r.get("sampling_aware", {}).get("all_auc", {}).get("estimate")
            w.writerow([r["clients"], r["config"], p and round(p["mu"], 3), p and round(p["epsilon"], 2),
                        p and round(p["auc_bound"], 3), round(r["accuracy_mean"], 4), round(c["all_auc"]["estimate"], 3),
                        round(c["all_auc"]["jackknife_975"][0], 3), round(c["all_auc"]["jackknife_975"][1], 3),
                        round(c["all_tpr_at_5pct"]["estimate"], 3), round(c["all_tpr_at_1pct"]["estimate"], 3),
                        round(c["top5_auc"]["estimate"], 3), sa and round(sa, 3), round(r["min_client_mean"], 3),
                        round(r["p10_client_mean"], 3), round(r["implied_mu"], 2)])
    with (folder / "matched.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clients", "comparison", "subset", "target", "mech_auc", "ref_auc", "difference", "low_975",
                    "high_975", "two_way_low", "two_way_high", "testability", "a1_outcome", "beats"])
        for n, comparisons in analysis["matched"].items():
            for name, block in comparisons.items():
                for subset in ("all", "top5"):
                    for r in block[subset]:
                        g = r["gate"]
                        w.writerow([n, name, subset, r["target"], _r(r["mechanism_auc"]), _r(r["reference_auc"]),
                                    _r(g["estimate"]), _r(g["low"]), _r(g["high"]), _r(r["two_way"]["low"]),
                                    _r(r["two_way"]["high"]), r["testability"], r.get("a1_outcome", ""), r.get("beats", "")])


def _r(value):
    return "" if value is None or value != value else round(value, 3)


if __name__ == "__main__":
    data = json.loads((HERE / "analysis.json").read_text())
    print(figure(data))
    tables(data)
