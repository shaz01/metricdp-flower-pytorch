"""Stage C privacy-utility frontiers: round-matched attack score vs late-training accuracy.

x = mean per-client IN/OUT attack score over targets (0.5 = no signal; per_client_score),
y = mean server accuracy of the IN trajectory over its last WINDOW rounds (R-WINDOW+1..R); under DP
the per-round accuracy swings by up to ~15 points on EuroSAT, so one round is not representative.
One PNG per dataset, one line per mechanism through its noise ratios, vanilla as the shared starting
point. x error bars span the per-target score range, y error bars the accuracy range over the window.

    python -m results.cia_frontier.dataset_vs_model.frontier [--root DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from results.cia_frontier.per_client_score import score_directories

ROOT = Path("results/cia_frontier/dataset_vs_model/results/stage_c")
WINDOW = 10
STYLE = {"global-dp": ("tab:blue", "o", "Global DP"), "metric-privacy": ("tab:orange", "s", "Metric privacy")}


def window_accuracy(metrics: dict, rounds: int, window: int = WINDOW) -> dict:
    """Mean/min/max server accuracy over rounds max(1, R-window+1)..R, plus the round-R value."""
    values = [float(metrics[str(r)]["accuracy"]) for r in range(max(1, rounds - window + 1), rounds + 1)]
    return {"accuracy": sum(values) / len(values), "accuracy_min": min(values), "accuracy_max": max(values),
            "accuracy_final": float(metrics[str(rounds)]["accuracy"]), "accuracy_window": len(values)}


def accuracies(root: Path, at_round: int | None = None) -> dict:
    """{(dataset, privacy, noise_ratio): window_accuracy(...)} from IN trajectories."""
    out = {}
    for path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest["out_target"] is not None:
            continue
        (run,) = [p for p in path.parent.glob(f"{manifest['run_name']}.json")]
        metrics = json.loads(run.read_text())["server_evaluate_metrics"]
        rounds = at_round or manifest["rounds"]
        out[(manifest["dataset"], manifest["privacy"], manifest["noise_ratio"])] = \
            window_accuracy(metrics, rounds)
    return out


def build(root: Path, at_round: int | None = None) -> dict:
    acc = accuracies(root, at_round)
    settings = []
    for entry in score_directories(root, at_round):
        key = (entry["dataset"], entry["privacy"], entry["noise_ratio"])
        scores = list(entry["per_target"].values())
        settings.append({"dataset": key[0], "privacy": key[1], "noise_ratio": key[2],
                         "rounds": at_round or entry["rounds"], **acc[key], "mean_score": entry["mean"],
                         "min_score": min(scores), "max_score": max(scores),
                         "per_target": entry["per_target"], "matched_rounds": entry["matched_rounds"]})
    settings.sort(key=lambda s: (s["dataset"], s["privacy"] != "vanilla", s["privacy"], s["noise_ratio"]))
    return {"x": "mean per-client round-matched IN/OUT score (0.5 = no signal)",
            "y": f"IN-trajectory server accuracy, mean over the last {WINDOW} rounds up to R",
            "error_bars": "x: per-target min-max; y: accuracy min-max over the window",
            "settings": settings}


def plot(data: dict, root: Path, out: Path | None = None) -> list[Path]:
    out = out or root
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for dataset in sorted({s["dataset"] for s in data["settings"]}):
        rows = [s for s in data["settings"] if s["dataset"] == dataset]
        (vanilla,) = [s for s in rows if s["privacy"] == "vanilla"]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for privacy, (color, marker, label) in STYLE.items():
            line = [vanilla] + sorted((s for s in rows if s["privacy"] == privacy), key=lambda s: s["noise_ratio"])
            xs, ys = [s["mean_score"] for s in line], [s["accuracy"] for s in line]
            err = [[s["mean_score"] - s["min_score"] for s in line], [s["max_score"] - s["mean_score"] for s in line]]
            yerr = [[s["accuracy"] - s["accuracy_min"] for s in line], [s["accuracy_max"] - s["accuracy"] for s in line]]
            ax.errorbar(xs, ys, xerr=err, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.8, capsize=0,
                        alpha=0.25, zorder=1)
            ax.plot(xs, ys, color=color, marker=marker, ms=7, lw=1.8, label=label, zorder=3)
            for s in line[1:]:
                ax.annotate(f"{s['noise_ratio']:g}", (s["mean_score"], s["accuracy"]), textcoords="offset points",
                            xytext=(5, -10), fontsize=7, color=color)
        ax.plot(vanilla["mean_score"], vanilla["accuracy"], "k*", ms=13, zorder=5, label="Vanilla")
        ax.axvline(0.5, color="grey", ls=":", lw=1)
        ax.set_xlabel("Mean per-client attack score (IN vs OUT, round-matched)")
        ax.set_ylabel(f"Accuracy, mean of rounds {vanilla['rounds'] - WINDOW + 1}-{vanilla['rounds']}")
        ax.set_title(f"Stage C frontier: {dataset}, Dirichlet α=0.3, seed 42 (bars: per-target / per-round min-max)",
                     fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out / f"frontier_{dataset}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--at-round", type=int, default=None,
                        help="Truncate to rounds 1..N (accuracy at N, attack over 1..N); writes to ROOT/at_round_N/")
    args = parser.parse_args()
    data = build(args.root, args.at_round)
    out = args.root / f"at_round_{args.at_round}" if args.at_round else args.root
    out.mkdir(exist_ok=True)
    (out / "frontier.json").write_text(json.dumps(data, indent=1, allow_nan=False) + "\n")
    for path in plot(data, args.root, out):
        print(path)


if __name__ == "__main__":
    main()
