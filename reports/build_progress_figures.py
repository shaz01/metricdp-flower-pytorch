"""Build the figures referenced by reports/progress_presentation.md.

Every number is read from committed result JSONs; nothing is hard-coded except the paper's own
Table 6 / Table 13 reference values, which are quoted from the source paper.

Run:
    uv run --with matplotlib python reports/build_progress_figures.py

Output: reports/figures/progress/*.png
"""

from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "reports" / "figures" / "progress"

METHODS = ("vanilla", "global-dp", "metric-privacy")
LABEL = {"vanilla": "Vanilla", "global-dp": "Global-DP", "metric-privacy": "Metric-privacy"}
COLOR = {"vanilla": "#444444", "global-dp": "#0072B2", "metric-privacy": "#D55E00"}
RATIOS = (0.0025, 0.004, 0.00625)

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titleweight": "bold",
    "figure.dpi": 150,
})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load(path: Path) -> dict:
    return json.loads(path.read_text())


def final_accuracy(run: dict) -> float:
    metrics = run["server_evaluate_metrics"]
    last = max(metrics, key=int)
    return float(metrics[last]["accuracy"])


def accuracy_curve(run: dict) -> tuple[list[int], list[float]]:
    metrics = run["server_evaluate_metrics"]
    rounds = sorted(metrics, key=int)
    return [int(r) for r in rounds], [float(metrics[r]["accuracy"]) for r in rounds]


def run_files(directory: Path, pattern: str = "*.json") -> list[Path]:
    return [
        p for p in sorted(directory.glob(pattern))
        if not p.name.endswith(".evaluation.json") and p.name not in {"cia.json", "colab_run.json", "planned_runs_manifest.json"}
    ]


def pooled_auc(rows: list[dict], field: str = "target_clean_shadow_loss") -> float:
    """All-IN-vs-all-OUT directional AUC on score = -loss, then let the attacker flip it."""
    ins = [-float(r[field]) for r in rows if "in-remove" in r["run_name"]]
    outs = [-float(r[field]) for r in rows if "out-remove" in r["run_name"]]
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in ins for b in outs)
    auc = wins / (len(ins) * len(outs))
    return max(auc, 1.0 - auc)


def round20_in_accuracy(directory: Path) -> float:
    in_run = next(p for p in run_files(directory) if "in-remove" in p.name)
    return float(load(in_run)["server_evaluate_metrics"]["20"]["accuracy"])


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}")


# ---------------------------------------------------------------------------
# Week 1: reproduction, ours vs paper
# ---------------------------------------------------------------------------

PAPER_TABLE6_FEDAVG = {"vanilla": 0.909, "global-dp": 0.884, "metric-privacy": 0.894}


def fig_w1_reproduction() -> None:
    directory = RESULTS / "planned_runs" / "reproduction" / "original_reproduction"
    ours: dict[str, list[float]] = {m: [] for m in METHODS}
    for path in run_files(directory):
        privacy = re.search(r"__(vanilla|global-dp|metric-privacy)__", path.name)[1]
        ours[privacy].append(final_accuracy(load(path)))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(METHODS))
    width = 0.36
    ax.bar([i - width / 2 for i in x], [PAPER_TABLE6_FEDAVG[m] for m in METHODS], width,
           color="#BBBBBB", label="Paper (Table 6)")
    ax.bar([i + width / 2 for i in x], [statistics.fmean(ours[m]) for m in METHODS], width,
           yerr=[statistics.stdev(ours[m]) for m in METHODS], capsize=4,
           color=[COLOR[m] for m in METHODS], label="Ours (3 seeds)")
    for i, m in enumerate(METHODS):
        ax.text(i - width / 2, PAPER_TABLE6_FEDAVG[m] + 0.01, f"{PAPER_TABLE6_FEDAVG[m]:.1%}", ha="center", fontsize=10)
        ax.text(i + width / 2, statistics.fmean(ours[m]) + 0.015, f"{statistics.fmean(ours[m]):.1%}", ha="center", fontsize=10)
    ax.set_xticks(list(x), [LABEL[m] for m in METHODS])
    ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Week 1 — Alzheimer, 4 clients, FedAvg, noise 0.01")
    ax.legend(loc="lower right", frameon=False)
    save(fig, "w1_reproduction.png")


# ---------------------------------------------------------------------------
# Week 2: more noise at 4 clients; noise sweep at 8 clients
# ---------------------------------------------------------------------------

def fig_w2_noise_sweep() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    # 4 clients: paper matrix + Appendix-B high-noise checks (global-DP FedAvg, seed 42)
    ax = axes[0]
    repro = RESULTS / "reproduce_paper" / "runs"
    points = {}
    for path in repro.glob("*__homogeneous__global-dp__fedavg__noise-*__seed-42.json"):
        nm = float(re.search(r"noise-([0-9.]+)", path.name)[1])
        points[nm] = final_accuracy(load(path))
    nms = sorted(points)
    ax.plot(nms, [points[n] for n in nms], "o-", color=COLOR["global-dp"], label="Global-DP")
    mp = {}
    for path in repro.glob("*__homogeneous__metric-privacy__fedavg__noise-*__seed-42.json"):
        nm = float(re.search(r"noise-([0-9.]+)", path.name)[1])
        mp[nm] = final_accuracy(load(path))
    ax.plot(sorted(mp), [mp[n] for n in sorted(mp)], "s-", color=COLOR["metric-privacy"], label="Metric-privacy")
    vanilla = final_accuracy(load(repro / "main__homogeneous__vanilla__fedavg__noise-0.01__seed-42.json"))
    ax.axhline(vanilla, ls="--", color=COLOR["vanilla"], label="Vanilla")
    ax.axhline(0.4953, ls=":", color="grey")
    ax.text(0.0035, 0.505, "majority-class guess", fontsize=9, color="grey")
    ax.set_xscale("log")
    ax.set_xlabel("Noise multiplier")
    ax.set_ylabel("Test accuracy")
    ax.set_title("4 clients: tied, then broken")
    ax.legend(frameon=False, loc="center left")

    # 8 clients: noise sweep (FedAvg, seed 42), completed runs only
    ax = axes[1]
    sweep = RESULTS / "noise_sweep"
    for privacy, marker in (("global-dp", "o"), ("metric-privacy", "s")):
        pts: dict[float, list[float]] = {}
        for path in sweep.glob(f"noise8__*__{privacy}__fedavg__nm*.json"):
            if path.name.endswith(".evaluation.json"):
                continue
            nm = float(re.search(r"nm([0-9p]+)", path.name)[1].replace("p", "."))
            pts.setdefault(nm, []).append(final_accuracy(load(path)))
        nms = sorted(pts)
        ax.plot(nms, [statistics.fmean(pts[n]) for n in nms], marker + "-", color=COLOR[privacy], label=LABEL[privacy])
    van = [final_accuracy(load(RESULTS / "8client_scaling" / f"scaling8__{p}__vanilla__fedavg.json")) for p in ("homogeneous", "non-iid")]
    ax.axhline(statistics.fmean(van), ls="--", color=COLOR["vanilla"], label="Vanilla")
    ax.axhline(0.4953, ls=":", color="grey")
    ax.annotate("gap appears\nat 0.05", xy=(0.05, 0.80), xytext=(0.014, 0.66), fontsize=10,
                arrowprops={"arrowstyle": "->"})
    ax.set_xscale("log")
    ax.set_xlabel("Noise multiplier")
    ax.set_title("8 clients: a gap, then broken")
    ax.legend(frameon=False, loc="lower left")
    fig.suptitle("Week 2 — accuracy vs. noise (Alzheimer, FedAvg; mean over both data splits)", y=1.02)
    save(fig, "w2_noise_sweep.png")


# ---------------------------------------------------------------------------
# Week 3: same run three times (MPS); advantage vs clients; formula; heatmap; Alzheimer attack
# ---------------------------------------------------------------------------

def fig_w3_same_run_thrice() -> None:
    archive = RESULTS / "archive" / "scale_controlled_mps_v1v2" / "scale_controlled"
    check = RESULTS / "noise_floor_check"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, privacy in zip(axes, ("global-dp", "metric-privacy")):
        runs = [archive / f"scalectrl__homogeneous__{privacy}__fedavg__n48.json",
                check / f"noisecheck__homogeneous__{privacy}__fedavg__n48__rep1.json",
                check / f"noisecheck__homogeneous__{privacy}__fedavg__n48__rep2.json"]
        for i, path in enumerate(runs):
            rounds, acc = accuracy_curve(load(path))
            ax.plot(rounds, acc, lw=1.4, alpha=0.9, label=f"run {i + 1}: final {acc[-1]:.0%}")
        ax.set_title(f"{LABEL[privacy]}, 48 clients")
        ax.set_xlabel("Round")
        ax.legend(frameon=False)
    axes[0].set_ylabel("Test accuracy")
    fig.suptitle("Week 3 — same settings, same seed, three runs on the Mac (MPS)", y=1.02)
    save(fig, "w3_same_run_thrice.png")


def fig_w3_advantage_vs_clients() -> None:
    def deltas(directory: Path, prefix: str, partition: str) -> tuple[list[int], list[float]]:
        ns, ds = [], []
        for n in (4, 8, 48):
            mp = directory / f"{prefix}__{partition}__metric-privacy__fedavg__n{n}.json"
            gdp = directory / f"{prefix}__{partition}__global-dp__fedavg__n{n}.json"
            if mp.exists() and gdp.exists():
                ns.append(n)
                ds.append(100 * (final_accuracy(load(mp)) - final_accuracy(load(gdp))))
        return ns, ds

    fig, ax = plt.subplots(figsize=(7, 4.2))
    cuda = RESULTS / "scale_controlled_epochs"
    for partition, color in (("homogeneous", "#009E73"), ("non-iid", "#CC79A7")):
        ns, ds = deltas(cuda, "scalectrlep", partition)
        ax.plot(ns, ds, "o-", color=color, label=f"GPU, fixed — {partition}")
    mps = RESULTS / "archive" / "scale_controlled_mps_v1v2" / "scale_controlled"
    ns, ds = deltas(mps, "scalectrl", "homogeneous")
    ax.plot(ns, ds, "x--", color="grey", label="Mac before the fix — homogeneous")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 48], ["4", "8", "48"])
    ax.set_xlabel("Number of clients")
    ax.set_ylabel("Metric-privacy − Global-DP (accuracy points)")
    ax.set_title("Week 3 — the advantage shrinks; it does not flip")
    ax.legend(frameon=False)
    save(fig, "w3_advantage_vs_clients.png")


def fig_w3_noise_ratio_formula() -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")
    ax.text(0.5, 0.85, r"Flower adds noise with standard deviation", ha="center", fontsize=14)
    ax.text(0.5, 0.62, r"$\sigma = \dfrac{nm \cdot C}{N}$", ha="center", fontsize=22)
    ax.text(0.5, 0.40, r"so a fixed $nm$ means less real noise as $N$ grows.", ha="center", fontsize=14)
    ax.text(0.5, 0.18, r"Set $nm = \mathrm{ratio} \cdot N$:   $\sigma = \dfrac{\mathrm{ratio}\cdot N \cdot C}{N} = \mathrm{ratio} \cdot C$",
            ha="center", fontsize=18)
    ax.text(0.5, -0.02, "nm = noise multiplier · C = clipping norm (5) · N = number of clients", ha="center", fontsize=10, color="grey")
    save(fig, "w3_noise_ratio_formula.png")


def fig_w3_heatmap() -> None:
    directory = RESULTS / "noise_by_clients"
    ns = (8, 16, 32, 48)
    nms = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, privacy in zip(axes, ("global-dp", "metric-privacy")):
        grid = []
        for n in ns:
            row = []
            for nm in nms:
                token = str(nm).replace(".", "p").rstrip("0").rstrip("p") if nm != 1.0 else "1"
                vals = []
                for partition in ("homogeneous", "non-iid"):
                    path = directory / f"noisebyclients__{partition}__{privacy}__fedavg__n{n}__nm{token}.json"
                    vals.append(final_accuracy(load(path)))
                row.append(statistics.fmean(vals))
            grid.append(row)
        im = ax.imshow(grid, cmap="viridis", vmin=0.3, vmax=1.0, aspect="auto")
        for i, n in enumerate(ns):
            for j, nm in enumerate(nms):
                ax.text(j, i, f"{grid[i][j]:.0%}", ha="center", va="center",
                        color="white" if grid[i][j] < 0.7 else "black", fontsize=10)
        ax.set_xticks(range(len(nms)), [str(x) for x in nms])
        ax.set_yticks(range(len(ns)), [str(x) for x in ns])
        ax.set_xlabel("Noise multiplier")
        ax.set_ylabel("Number of clients")
        ax.set_title(LABEL[privacy])
    fig.colorbar(im, ax=axes, label="Test accuracy", shrink=0.9)
    fig.suptitle("Week 3 — the noise level where training breaks rises with client count (Alzheimer, FedAvg)", y=1.02)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "w3_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / 'w3_heatmap.png'}")


PAPER_TABLE13_NOISY_AUC = {"vanilla": 0.890, "global-dp": 0.397, "metric-privacy": 0.493}


def fig_w3_alzheimer_attack() -> None:
    base = RESULTS / "planned_runs" / "alzheimer"
    rows = load(base / "in_remove" / "alzheimer-in-remove" / "cia.json") + load(base / "out_remove" / "alzheimer-out-remove" / "cia.json")
    ours = {}
    for m in METHODS:
        subset = [r for r in rows if r["privacy"] == m]
        ins = [-float(r["target_noisy_shadow_loss"]) for r in subset if "in-remove" in r["run_name"]]
        outs = [-float(r["target_noisy_shadow_loss"]) for r in subset if "out-remove" in r["run_name"]]
        ours[m] = sum(1.0 if a > b else 0.5 if a == b else 0.0 for a in ins for b in outs) / (len(ins) * len(outs))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(METHODS))
    width = 0.36
    ax.bar([i - width / 2 for i in x], [PAPER_TABLE13_NOISY_AUC[m] for m in METHODS], width, color="#BBBBBB", label="Paper (Table 13)")
    ax.bar([i + width / 2 for i in x], [ours[m] for m in METHODS], width, color=[COLOR[m] for m in METHODS], label="Ours (3 seeds pooled)")
    for i, m in enumerate(METHODS):
        ax.text(i - width / 2, PAPER_TABLE13_NOISY_AUC[m] + 0.02, f"{PAPER_TABLE13_NOISY_AUC[m]:.0%}", ha="center", fontsize=10)
        ax.text(i + width / 2, ours[m] + 0.02, f"{ours[m]:.0%}", ha="center", fontsize=10)
    ax.axhline(0.5, ls=":", color="grey")
    ax.text(2.45, 0.515, "coin flip", fontsize=9, color="grey", ha="right")
    ax.set_xticks(list(x), [LABEL[m] for m in METHODS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Attacker success (AUC)")
    ax.set_title("Week 3 — Alzheimer, 3 clients, noisy shadow: paper vs. ours")
    ax.legend(frameon=False, loc="lower left")
    save(fig, "w3_alzheimer_attack.png")


# ---------------------------------------------------------------------------
# Week 4: attack vs clients; ratio sweep at 8 clients
# ---------------------------------------------------------------------------

def fig_w4_attack_vs_clients() -> None:
    rows = load(RESULTS / "cia" / "cifar10_remove" / "cia_analysis.json")
    by: dict[str, dict[int, float]] = {m: {} for m in METHODS}
    acc: dict[str, dict[int, float]] = {m: {} for m in METHODS}
    for r in rows:
        auc = r["multi_round"]["clean"]["pooled_auc"]
        by[r["privacy"]][int(r["num_clients_canonical"])] = max(auc, 1 - auc)
        acc[r["privacy"]][int(r["num_clients_canonical"])] = r["utility"]["in"]["accuracy_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for m in METHODS:
        ns = sorted(by[m])
        axes[0].plot(ns, [by[m][n] for n in ns], "o-", color=COLOR[m], label=LABEL[m])
        axes[1].plot(ns, [acc[m][n] for n in ns], "o-", color=COLOR[m], label=LABEL[m])
    axes[0].axhline(0.5, ls=":", color="grey")
    axes[0].text(100, 0.515, "coin flip", fontsize=9, color="grey", ha="right")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].set_ylabel("Attacker success (AUC)")
    axes[0].set_title("Attack gets harder with more clients")
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_title("Accuracy, same runs")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 48, 100], ["8", "16", "48", "100"])
        ax.set_xlabel("Number of clients")
        ax.legend(frameon=False)
    fig.suptitle("Week 4 — CIFAR-10, noise multiplier 0.0182, clean shadow", y=1.02)
    save(fig, "w4_attack_vs_clients.png")


def ratio_sweep_dir(clients: int, privacy: str, ratio: float | None) -> Path:
    if privacy == "vanilla":
        name = f"n{clients}-vanilla"
    else:
        short = {"global-dp": "gdp", "metric-privacy": "mp"}[privacy]
        name = f"n{clients}-{short}-r{str(ratio).replace('.', 'p')}"
    return RESULTS / "cia" / "cifar10_remove_ratio_sweep" / name / "runs"


def fig_w4_ratio_sweep(clients: int = 8) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = [0.0, *RATIOS]
    for privacy in ("global-dp", "metric-privacy"):
        accs, aucs = [], []
        for ratio in x:
            d = ratio_sweep_dir(clients, "vanilla" if ratio == 0 else privacy, None if ratio == 0 else ratio)
            accs.append(round20_in_accuracy(d))
            aucs.append(pooled_auc(load(d / "cia.json")))
        axes[0].plot(range(len(x)), accs, "o-", color=COLOR[privacy], label=LABEL[privacy])
        axes[1].plot(range(len(x)), aucs, "o-", color=COLOR[privacy], label=LABEL[privacy])
        for i, (a, u) in enumerate(zip(accs, aucs)):
            axes[0].text(i, a + 0.012, f"{a:.0%}", ha="center", fontsize=9, color=COLOR[privacy])
            axes[1].text(i, u + 0.012, f"{u:.0%}", ha="center", fontsize=9, color=COLOR[privacy])
    ticks = ["none\n(vanilla)", *[str(r) for r in RATIOS]]
    axes[0].set_ylabel("Test accuracy (round 20)")
    axes[0].set_title("Accuracy: metric-privacy holds, global-DP breaks")
    axes[1].set_ylabel("Attacker success (AUC)")
    axes[1].set_title("Attacker: first noise helps, more does not")
    axes[1].axhline(0.5, ls=":", color="grey")
    axes[1].set_ylim(0.4, 1.0)
    for ax in axes:
        ax.set_xticks(range(len(x)), ticks)
        ax.set_xlabel("Noise ratio")
        ax.legend(frameon=False, loc="lower left")
    fig.suptitle(f"Week 4 — CIFAR-10, {clients} clients, clean shadow", y=1.02)
    save(fig, f"w4_ratio_sweep_{clients}clients.png")


# ---------------------------------------------------------------------------
# Week 5: Dirichlet sweep
# ---------------------------------------------------------------------------

def dirichlet_dirs() -> dict[tuple[float, str, float], Path]:
    base = RESULTS / "dirichlet" / "cifar10" / "8_clients"
    out = {}
    for d in base.iterdir():
        m = re.fullmatch(r"alpha-([0-9p]+)__(vanilla|global-dp|metric-privacy)__.*__noise-([0-9p]+)", d.name)
        if m and (d / "cia.json").exists():
            out[(float(m[1].replace("p", ".")), m[2], float(m[3].replace("p", ".")))] = d
    return out


def fig_w5_dirichlet() -> None:
    dirs = dirichlet_dirs()
    alphas = (0.1, 1.5, 3.0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for col, alpha in enumerate(alphas):
        for privacy in ("global-dp", "metric-privacy"):
            xs, accs, aucs = [], [], []
            vanilla = next((d for (a, p, r), d in dirs.items() if a == alpha and p == "vanilla"), None)
            if vanilla is not None:
                xs.append(0); accs.append(round20_in_accuracy(vanilla)); aucs.append(pooled_auc(load(vanilla / "cia.json")))
            for i, ratio in enumerate(RATIOS, start=1):
                d = dirs.get((alpha, privacy, ratio))
                if d is None:
                    continue
                xs.append(i); accs.append(round20_in_accuracy(d)); aucs.append(pooled_auc(load(d / "cia.json")))
            axes[0][col].plot(xs, accs, "o-", color=COLOR[privacy], label=LABEL[privacy])
            axes[1][col].plot(xs, aucs, "o-", color=COLOR[privacy], label=LABEL[privacy])
        label = {0.1: "α = 0.1 (extreme skew)", 1.5: "α = 1.5", 3.0: "α = 3 (mild skew)"}[alpha]
        axes[0][col].set_title(label)
        axes[1][col].axhline(0.5, ls=":", color="grey")
        axes[1][col].set_xticks(range(4), ["none", *[str(r) for r in RATIOS]])
        axes[1][col].set_xlabel("Noise ratio")
        axes[0][col].set_ylim(0.2, 0.8)
        axes[1][col].set_ylim(0.4, 1.0)
    axes[0][0].set_ylabel("Test accuracy (round 20)")
    axes[1][0].set_ylabel("Attacker success (AUC)")
    axes[0][0].legend(frameon=False, loc="lower left")
    fig.suptitle("Week 5 — CIFAR-10, 8 clients, Dirichlet label skew (point at 'none' = vanilla)", y=1.0)
    save(fig, "w5_dirichlet.png")


def fig_w5_distance() -> None:
    dirs = dirichlet_dirs()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    style = {0.1: "#CC0000", 1.5: "#E69F00", 3.0: "#009E73"}
    for alpha in (0.1, 1.5, 3.0):
        d = dirs.get((alpha, "metric-privacy", 0.00625)) or dirs.get((alpha, "metric-privacy", 0.004))
        in_run = next(p for p in run_files(d) if "in-remove" in p.name)
        tm = load(in_run)["train_metrics"]
        rounds = sorted(tm, key=int)
        ax.plot([int(r) for r in rounds], [tm[r]["metric-dp-distance"] for r in rounds], "-", color=style[alpha], lw=2,
                label=f"α = {alpha:g}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Max pairwise client-model distance  d")
    ax.set_title("Week 5 — more skew → larger d → less noise (noise = nm ÷ d)")
    ax.legend(frameon=False)
    save(fig, "w5_distance.png")


if __name__ == "__main__":
    fig_w1_reproduction()
    fig_w2_noise_sweep()
    fig_w3_same_run_thrice()
    fig_w3_advantage_vs_clients()
    fig_w3_noise_ratio_formula()
    fig_w3_heatmap()
    fig_w3_alzheimer_attack()
    fig_w4_attack_vs_clients()
    fig_w4_ratio_sweep(8)
    fig_w5_dirichlet()
    fig_w5_distance()
