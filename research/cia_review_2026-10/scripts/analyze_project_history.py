"""Re-derive the quantitative project history used by research/cia_review_2026-10/.

Analysis of committed result files only -- no training, no new runs.
Run from the repository root:

    uv run --with pandas --with matplotlib python research/cia_review_2026-10/scripts/analyze_project_history.py

Outputs (all under research/cia_review_2026-10/):
    data/results_history.csv          one row per headline result, with its source file
    data/eurosat_frontier_scores.csv  per (mechanism, seed, target) IN/OUT shadow-loss scores, clean and noisy view
    data/eurosat_frontier_gap_by_round.csv
    data/metric_noise_3client.csv     per-round IN vs OUT metric-privacy noise std (3-client plan suite)
    data/metric_noise_48client_counterfactual.csv  one-round leave-one-out noise-scale shift per client (EuroSAT, 48 clients)
    figures/results_history.png, figures/client_noise_vs_baseline.png, figures/metric_noise_side_channel.png

Every number is computed from the files named in the `source` column; the figure captions in
project_state.md describe exactly what is plotted.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("REPO_ROOT", os.getcwd())
OUT = os.path.join(ROOT, "research", "cia_review_2026-10")
FR = os.path.join(ROOT, "results/cia_frontier/eurosat_frontier/results/frontier")
PS = os.path.join(ROOT, "results/contest_at_scale/plan_suite/results")
SH = os.path.join(ROOT, "results/stacked_head")
AUDIT = os.path.join(ROOT, "research/project_evidence_audit.md")
INFL = os.path.join(ROOT, "results/cia_frontier/influence_noise/README.md")

COL = {"global-dp": "#4C72B0", "metric-privacy": "#DD8452", "stacked head (client noise)": "#55A868",
       "vanilla": "#7F7F7F"}


def rel(p: str) -> str:
    return os.path.relpath(p, ROOT)


def style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False, "axes.spines.right": False,
        "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
        "savefig.dpi": 300, "figure.dpi": 150, "axes.titlelocation": "left",
    })


# ---------------------------------------------------------------- loaders
def load_train(run_dir: str):
    ps = [p for p in glob.glob(f"{run_dir}/*/*.json")
          if not p.endswith(("evaluation.json", "provenance.json", "complete.json", "manifest.json",
                             "partitions.json", "measurements.json"))]
    return json.load(open(ps[0])) if ps else None


def load_meas(run_dir: str):
    ps = glob.glob(f"{run_dir}/*/measurements.json")
    return pd.DataFrame(json.load(open(ps[0]))) if ps else None


# ---------------------------------------------------------------- 1. AUC-frontier landings (audit table)
def frontier_landings() -> pd.DataFrame:
    txt = open(AUDIT).read()
    block = txt.split("### Landing values recomputed from raw state files", 1)[1]
    rows = []
    for line in block.splitlines():
        m = re.match(r"\|\s*(Alzheimer|CIFAR-10|EuroSAT|Fashion-MNIST)\s*\|\s*([\w-]+)\s*\|\s*([\w-]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            rows.append(dict(dataset=m[1], partition=m[2],
                             mechanism="global-dp" if m[3] == "global-DP" else "metric-privacy",
                             accuracy_pct=float(m[4]), folded_score_42_44=float(m[5]),
                             folded_score_43_44=float(m[6])))
        elif rows and not line.startswith("|"):
            break
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2. EuroSAT frontier per-target scores
def eurosat_scores():
    runs = sorted(os.listdir(FR))
    rec = []
    for mode in ("global-dp", "metric-privacy"):
        for seed in (42, 43, 44):
            rin = [r for r in runs if r.startswith(mode) and r.endswith(f"seed-{seed}-in")]
            if not rin:
                continue
            mi = load_meas(os.path.join(FR, rin[0]))
            for t in range(10):
                rout = [r for r in runs if r.startswith(mode) and r.endswith(f"seed-{seed}-out-{t}")]
                if not rout:
                    continue
                mo = load_meas(os.path.join(FR, rout[0]))
                a = mi[mi.target == t].set_index("round")
                b = mo[mo.target == t].set_index("round")
                j = a.join(b, lsuffix="_in", rsuffix="_out", how="inner")
                for r_, row in j.iterrows():
                    rec.append(dict(mechanism=mode, seed=seed, target=t, round=int(r_),
                                    gap_clean=row.clean_loss_out - row.clean_loss_in,
                                    gap_noisy=row.noisy_loss_out - row.noisy_loss_in))
    lk = pd.DataFrame(rec)
    sc = (lk.groupby(["mechanism", "seed", "target"])
          .agg(score_clean=("gap_clean", lambda x: float((x > 0).mean())),
               score_noisy=("gap_noisy", lambda x: float((x > 0).mean())),
               n_rounds=("round", "size")).reset_index())
    return lk, sc


# ---------------------------------------------------------------- 3. metric-privacy noise side channel
def metric_noise_3client() -> pd.DataFrame:
    rows = []
    for ds in sorted(os.listdir(PS)):
        for seed in (42, 43, 44):
            fin = [f for f in glob.glob(f"{PS}/{ds}/in_remove/*/*__metric-privacy__fedavg__clients-3__seed-{seed}__*.json")
                   if not f.endswith("evaluation.json")]
            fout = [f for f in glob.glob(f"{PS}/{ds}/out_remove/*/*__metric-privacy__fedavg__clients-2__seed-{seed}__*.json")
                    if not f.endswith("evaluation.json")]
            if not fin or not fout:
                continue
            A = json.load(open(fin[0]))["train_metrics"]
            B = json.load(open(fout[0]))["train_metrics"]
            for r in sorted(A, key=int):
                a, b = A[r], B.get(r)
                if b is None or "metric-dp-distance" not in a:
                    continue
                d = np.asarray(a.get("metric-dp-pairwise-distances", []), float)
                ii = np.asarray(a.get("metric-dp-pairwise-client-i", []))
                jj = np.asarray(a.get("metric-dp-pairwise-client-j", []))
                k = int(np.argmax(d))
                rows.append(dict(dataset=ds, seed=seed, round=int(r), d_in=a["metric-dp-distance"],
                                 d_out=b["metric-dp-distance"], sd_in=a["metric-dp-noise-stdv"],
                                 sd_out=b["metric-dp-noise-stdv"],
                                 target_in_max_pair=int(2 in (int(ii[k]), int(jj[k]))),
                                 source_in=rel(fin[0]), source_out=rel(fout[0])))
    df = pd.DataFrame(rows)
    df["sd_ratio_in_over_out"] = df.sd_in / df.sd_out
    df["d_ratio_in_over_out"] = df.d_in / df.d_out
    return df


def metric_noise_48client() -> pd.DataFrame:
    out = []
    for seed in (42, 43, 44):
        J = load_train(os.path.join(FR, f"metric-privacy-r0.001546-seed-{seed}-in"))
        for r, a in J["train_metrics"].items():
            d = np.asarray(a["metric-dp-pairwise-distances"], float)
            ii = np.asarray(a["metric-dp-pairwise-client-i"])
            jj = np.asarray(a["metric-dp-pairwise-client-j"])
            if len(d) == 0 or not np.isfinite(d).all():
                continue
            dmax = d.max()
            for x in np.unique(np.concatenate([ii, jj])):
                m = (ii != x) & (jj != x)
                # sigma_t = ratio*C/d_t, so log(sigma_without_x / sigma_with_x) = log(d_with / d_without)
                out.append(dict(seed=seed, round=int(r), client=int(x), log_noise_shift=float(np.log(dmax / d[m].max()))))
    return pd.DataFrame(out)


# ---------------------------------------------------------------- 4. stacked-head comparisons
def stacked_head():
    R1 = json.load(open(os.path.join(SH, "comparison_part1_report.json")))
    R2 = json.load(open(os.path.join(SH, "comparison_part2_report.json")))
    p1 = []
    for base, v in R1["baselines"].items():
        for g in v["groups"]:
            p1.append(dict(baseline=base, task=g["task"], risk=g["risk"], n_sets=g["n_sets"],
                           diff=g["mean_difference"], lo=g["ci95"][0], hi=g["ci95"][1], verdict=g["verdict"],
                           ours_gain=g["ours_mean_gain"], baseline_gain=g["baseline_mean_gain"],
                           baseline_realized_auc=g["baseline_realized_oracle_auc"]))
    p2 = []
    for base, v in R2["baselines"].items():
        for task, tv in v["tasks"].items():
            for lvl, (auc, gain) in tv["baseline_points"].items():
                p2.append(dict(series=base, task=task, level=lvl, practical_auc=auc, gain=gain))
            if base == "global-dp":  # ours is identical in both baseline blocks
                for c in tv["comparisons"]:
                    p2.append(dict(series="stacked head (client noise)", task=task, level=c["risk"],
                                   practical_auc=c["ours_auc"], gain=c["ours_gain"]))
    return pd.DataFrame(p1), pd.DataFrame(p2)


def influence_noise() -> pd.DataFrame:
    txt = open(INFL).read()
    m = re.search(r"f = 0\.05 costs ([\d.]+) points of accuracy against the control \(([\d.]+)% vs ([\d.]+)%\)\s+and\s+doesn't lower the attack score \(([\d.]+) vs ([\d.]+)\)", txt.replace("\n", " "))
    proto = open(INFL.replace("README.md", "PROTOCOL.md")).read()
    m2 = re.search(r"IN accuracy\s+([\d.]+)% vs ([\d.]+)%", proto)  # f = .5 IN vs f = 0 IN (server final test)
    rows = [dict(arm="f=0 (isotropic control)", accuracy_pct=float(m[3]), attack_score=float(m[5])),
            dict(arm="f=0.05", accuracy_pct=float(m[2]), attack_score=float(m[4])),
            dict(arm="f=0.5 (stopped)", accuracy_pct=float(m2[1]), attack_score=np.nan)]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures
def fig_results_history(land, sc, lk, p2, path):
    fig, axs = plt.subplots(2, 2, figsize=(7.2, 5.6))
    # (a) frontier landings
    ax = axs[0, 0]
    marks = {"Alzheimer": "s", "CIFAR-10": "o", "EuroSAT": "^", "Fashion-MNIST": "D"}
    for (ds, part), g in land.groupby(["dataset", "partition"]):
        g = g.sort_values("mechanism")
        if len(g) == 2:
            ax.plot(g.folded_score_42_44, g.accuracy_pct, color="0.8", lw=0.8, zorder=1)
        for _, r in g.iterrows():
            ax.scatter(r.folded_score_42_44, r.accuracy_pct, marker=marks[ds], s=26, color=COL[r.mechanism],
                       edgecolor="k" if part == "non-iid" else "none", lw=0.6, zorder=2)
    ax.axvline(0.5, color="0.6", ls=":", lw=0.8)
    for ds, mk in marks.items():
        ax.scatter([], [], marker=mk, color="0.5", s=20, label=ds)
    ax.scatter([], [], marker="o", color=COL["global-dp"], s=20, label="global DP")
    ax.scatter([], [], marker="o", color=COL["metric-privacy"], s=20, label="metric privacy")
    ax.scatter([], [], marker="o", facecolor="none", edgecolor="k", s=20, label="black edge = non-IID")
    ax.legend(loc="lower left", fontsize=6, ncol=1, handletextpad=0.3)
    ax.set_xlabel("folded paired CIA score, seeds 42-44 (0.5 = chance)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("AUC-targeted landings: neither mechanism dominates")
    ax.margins(0.06)
    # (b) per-target spread
    ax = axs[0, 1]
    rng = np.random.default_rng(0)
    for i, mech in enumerate(("global-dp", "metric-privacy")):
        v = sc[sc.mechanism == mech].score_clean.values
        x = i + rng.uniform(-0.12, 0.12, len(v))
        ax.scatter(x, v, s=10, color=COL[mech], alpha=0.8)
        ax.hlines(np.median(v), i - 0.25, i + 0.25, color="k", lw=1.2)
    ax.axhline(0.5, color="0.6", ls=":", lw=0.8)
    ax.set_xticks([0, 1], ["global DP", "metric privacy"])
    ax.set_ylabel("per-target CIA score (clean view)")
    ax.set_title("EuroSAT, 48 clients: per-client leakage varies widely")
    ax.margins(x=0.25, y=0.06)
    # (c) stacked-head frontier
    ax = axs[1, 0]
    for task, ls in (("kmnist_classes0to3", "-"), ("kmnist_classes4to7", "--")):
        for ser in ("global-dp", "metric-privacy", "stacked head (client noise)"):
            g = p2[(p2.task == task) & (p2.series == ser) & (p2.level != "nf")]
            order = ["q55", "q65", "q80", "q90", "q95"]
            g = g.set_index("level").loc[[o for o in order if o in g.level.values]]
            ax.plot(g.practical_auc, g.gain, ls=ls, marker="o", ms=3, color=COL[ser], lw=1)
    for ser in ("global-dp", "metric-privacy", "stacked head (client noise)"):
        ax.plot([], [], color=COL[ser], label={"global-dp": "global DP", "metric-privacy": "metric privacy"}.get(ser, ser))
    ax.plot([], [], color="0.4", ls="-", label="KMNIST 0-3"); ax.plot([], [], color="0.4", ls="--", label="KMNIST 4-7")
    ax.legend(loc="lower right", fontsize=6)
    ax.set_xlabel("practical model-only CIA AUC")
    ax.set_ylabel("held-out CE gain over public-only")
    ax.set_title("Head-step study: metric privacy tracks global DP")
    ax.margins(0.06)
    # (d) gap growth over rounds
    ax = axs[1, 1]
    for mech in ("global-dp", "metric-privacy"):
        g = lk[lk.mechanism == mech].groupby("round").gap_clean
        mu, q1, q3 = g.mean(), g.quantile(0.25), g.quantile(0.75)
        ax.fill_between(mu.index, q1, q3, color=COL[mech], alpha=0.18, lw=0)
        ax.plot(mu.index, mu.values, color=COL[mech], lw=1.2,
                label={"global-dp": "global DP", "metric-privacy": "metric privacy"}[mech])
    ax.axhline(0, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("FL round")
    ax.set_ylabel("OUT minus IN shadow loss (clean view)")
    ax.set_title("Clean-view IN/OUT gap is largest in rounds 1-25")
    ax.legend(loc="upper left", fontsize=6)
    for a, l in zip(axs.flat, "abcd"):
        a.text(-0.14, 1.06, l, transform=a.transAxes, fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_client_noise(p1, infl, path):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw=dict(width_ratios=[1.7, 1]))
    ax = axs[0]
    labs = []
    groups = p1[p1.baseline == "global-dp"][["task", "risk"]].values.tolist()
    for k, (task, risk) in enumerate(groups):
        for off, base in ((-0.15, "global-dp"), (0.15, "metric-privacy")):
            r = p1[(p1.baseline == base) & (p1.task == task) & (p1.risk == risk)].iloc[0]
            ax.errorbar(r["diff"], k + off, xerr=[[r["diff"] - r.lo], [r.hi - r["diff"]]], fmt="o", ms=3,
                        color=COL[base], lw=1, capsize=0)
        labs.append(f"{task.replace('_classes', ' ').replace('to', '-').replace('fmnist', 'F-MNIST').replace('kmnist', 'KMNIST').replace('mnist', 'MNIST')}  risk {risk:.2f}")
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_yticks(range(len(groups)), labs)
    ax.invert_yaxis()
    ax.set_xlabel("CE gain difference: client-noise construction minus baseline (95% CI)")
    ax.errorbar([], [], fmt="o", color=COL["global-dp"], label="vs global DP")
    ax.errorbar([], [], fmt="o", color=COL["metric-privacy"], label="vs metric privacy")
    ax.legend(loc="lower left", fontsize=6)
    ax.set_title("Risk 0.80: server mechanisms win every point estimate")
    ax.text(0.02, 0.98, "right of 0 = client noise better", transform=ax.transAxes, ha="left", va="top", fontsize=6)
    ax = axs[1]
    x = np.arange(len(infl))
    ax.bar(x, infl.accuracy_pct, color=["0.6", "#8172B3", "#8172B3"], width=0.6)
    for xi, (acc, s) in enumerate(zip(infl.accuracy_pct, infl.attack_score)):
        ax.text(xi, acc + 1, f"{acc:.1f}%" + (f"\nscore {s:.2f}" if np.isfinite(s) else "\nstopped"), ha="center", fontsize=6)
    ax.set_xticks(x, ["f=0\nisotropic", "f=0.05", "f=0.5"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("IN accuracy (%)")
    ax.set_title("Influence-directed noise\ncosts accuracy")
    for a, l in zip(axs, "ab"):
        a.text(-0.08 if l == "a" else -0.3, 1.06, l, transform=a.transAxes, fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_side_channel(m3, m48, path):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axs[0]
    rng = np.random.default_rng(1)
    dss = sorted(m3.dataset.unique())
    for i, ds in enumerate(dss):
        g = m3[m3.dataset == ds]
        ax.scatter(i - 0.12 + rng.uniform(-0.07, 0.07, len(g)), g.sd_ratio_in_over_out, s=6, color=COL["metric-privacy"], alpha=0.7)
        ax.scatter(i + 0.12 + rng.uniform(-0.07, 0.07, len(g)), 1 / g.d_ratio_in_over_out * (2 / 3), s=6, color="0.55", alpha=0.6)
    ax.axhline(2 / 3, color="k", ls=":", lw=0.8)
    ax.text(len(dss) - 0.5, 2 / 3 + 0.01, "client-count effect alone (2/3)", ha="right", va="bottom", fontsize=6)
    ax.scatter([], [], color=COL["metric-privacy"], s=10, label="observed sigma_IN / sigma_OUT")
    ax.scatter([], [], color="0.55", s=10, label="(2/3) x d_OUT / d_IN")
    ax.legend(loc="lower left", fontsize=6)
    ax.set_xticks(range(len(dss)), [{"alzheimer": "Alzheimer", "cifar": "CIFAR-10", "fashion": "Fashion-MNIST"}.get(d, d) for d in dss])
    ax.set_ylabel("metric-privacy noise std, IN / OUT")
    ax.set_title("3-client runs: IN noise is lower in every round")
    ax.margins(x=0.15)
    ax = axs[1]
    pc = m48.groupby(["seed", "client"]).log_noise_shift.agg(mean="mean", frac=lambda x: float((x > 0).mean())).reset_index()
    for seed, mk in zip((42, 43, 44), "os^"):
        g = pc[pc.seed == seed]
        ax.scatter(g.frac, 100 * g["mean"], s=10, marker=mk, color=COL["metric-privacy"], alpha=0.7, label=f"seed {seed}")
    ax.set_xlabel("share of rounds the client is in the max-distance pair")
    ax.set_ylabel("mean noise-std increase if removed (%)")
    ax.set_title("48 clients: only outlier clients move the noise")
    ax.legend(loc="upper left", fontsize=6)
    ax.margins(0.06)
    for a, l in zip(axs, "ab"):
        a.text(-0.14, 1.06, l, transform=a.transAxes, fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    style()
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    land = frontier_landings()
    lk, sc = eurosat_scores()
    m3 = metric_noise_3client()
    m48 = metric_noise_48client()
    p1, p2 = stacked_head()
    infl = influence_noise()

    hist = []
    for _, r in land.iterrows():
        hist.append(dict(study="AUC-targeted noise sweep (landed points)", setting=f"{r.dataset} {r.partition}",
                         mechanism=r.mechanism, metric="accuracy_pct | folded paired score (seeds 42-44)",
                         value=f"{r.accuracy_pct:.2f} | {r.folded_score_42_44:.3f}", source=rel(AUDIT)))
    for (mech, seed), g in sc.groupby(["mechanism", "seed"]):
        hist.append(dict(study="EuroSAT per-client frontier (alpha 0.3, 48 clients, nr 0.001546)",
                         setting=f"seed {seed}, {len(g)} targets", mechanism=mech,
                         metric="mean per-target score, clean | noisy shadow view",
                         value=f"{g.score_clean.mean():.3f} | {g.score_noisy.mean():.3f}",
                         source=rel(FR) + "/*/measurements.json"))
    for _, r in infl.iterrows():
        hist.append(dict(study="Influence-directed noise pilot (EuroSAT)", setting=r.arm, mechanism="influence-noise",
                         metric="IN accuracy_pct | attack score", value=f"{r.accuracy_pct:.1f} | {r.attack_score:.2f}",
                         source=rel(INFL)))
    for _, r in p1.iterrows():
        hist.append(dict(study="Stacked head vs paper mechanisms, Part 1 (matched calibrated risk)",
                         setting=f"{r.task} risk {r.risk}", mechanism=f"ours minus {r.baseline}",
                         metric="mean CE-gain difference [95% CI] | baseline realized oracle AUC",
                         value=f"{r['diff']:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}] | {r.baseline_realized_auc:.3f}",
                         source=rel(os.path.join(SH, 'comparison_part1_report.json'))))
    for ds, g in m3.groupby("dataset"):
        hist.append(dict(study="Metric-privacy noise level, 3-client CIA plan suite (nm 0.01, fixed)",
                         setting=f"{ds}, 3 seeds x 20 rounds", mechanism="metric-privacy",
                         metric="mean sigma_IN/sigma_OUT | mean d_IN/d_OUT | share rounds target in max pair",
                         value=f"{g.sd_ratio_in_over_out.mean():.3f} | {g.d_ratio_in_over_out.mean():.3f} | {g.target_in_max_pair.mean():.3f}",
                         source=rel(PS) + f"/{ds}/(in|out)_remove"))
    pc = m48.groupby(["seed", "client"]).log_noise_shift.mean()
    for seed in (42, 43, 44):
        s = pc.loc[seed]
        hist.append(dict(study="Metric-privacy leave-one-out noise shift, EuroSAT 48 clients (one-round counterfactual)",
                         setting=f"seed {seed}", mechanism="metric-privacy",
                         metric="max per-client mean log shift | share of clients with mean shift > 1%",
                         value=f"{s.max():.4f} | {(s > 0.01).mean():.3f}",
                         source=rel(FR) + f"/metric-privacy-r0.001546-seed-{seed}-in"))
    pd.DataFrame(hist).to_csv(os.path.join(OUT, "data/results_history.csv"), index=False)
    sc.to_csv(os.path.join(OUT, "data/eurosat_frontier_scores.csv"), index=False)
    lk.groupby(["mechanism", "round"]).agg(mean_gap_clean=("gap_clean", "mean"), mean_gap_noisy=("gap_noisy", "mean"),
                                         n=("gap_clean", "size")).reset_index().to_csv(
        os.path.join(OUT, "data/eurosat_frontier_gap_by_round.csv"), index=False)
    m3.to_csv(os.path.join(OUT, "data/metric_noise_3client.csv"), index=False)
    m48.groupby(["seed", "client"]).log_noise_shift.agg(mean_log_shift="mean", max_log_shift="max",
                                                         share_rounds_in_max_pair=lambda x: float((x > 0).mean())).reset_index().to_csv(
        os.path.join(OUT, "data/metric_noise_48client_counterfactual.csv"), index=False)

    fig_results_history(land, sc, lk, p2, os.path.join(OUT, "figures/results_history.png"))
    fig_client_noise(p1, infl, os.path.join(OUT, "figures/client_noise_vs_baseline.png"))
    fig_side_channel(m3, m48, os.path.join(OUT, "figures/metric_noise_side_channel.png"))
    return dict(land=land, sc=sc, lk=lk, m3=m3, m48=m48, p1=p1, p2=p2, infl=infl)


if __name__ == "__main__":
    main()
