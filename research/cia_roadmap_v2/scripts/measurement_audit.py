"""Measurement audit of the 2026-10 CIA programme, from committed result files only (no training).

Three checks:
  A. Do the IN and OUT trajectories of one seed share the server's DP noise draws?  Statistic per round:
     r_t = ||Delta_t||^2 - ||s_t||^2 - E||xi_t||^2 = (||xi_t||^2 - sigma^2 D) + 2 <s_t, xi_t>, from the released-update
     norm (releases.json) and the logged clipped-signal norm and expected noise norm (training JSON). Its correlation
     between two trajectories is about 1 if they draw the same noise vectors and about 0 if they draw independent ones.
  B. How much of the paired clean-view score would survive if the two worlds drew independent noise?  Upper bound that
     adds one round of independent noise only: per round P(g + e > 0), e ~ N(0, 2 sigma^2 G^2), G the logged
     shadow-gradient norm (accumulated noise from earlier rounds would lower the score further).
  C. Without DP, how much does sharing the training seed between IN and OUT inflate the paired score?  WP2 seeds 42-45
     share one partition layout, so IN of seed a can be compared with OUT of seed b != a.

Outputs (research/cia_roadmap_v2/): tables/noise_coupling_correlations.csv, tables/decoupled_score_bound.csv,
tables/cross_seed_scores_wp2.csv, tables/label_skew_vs_exposure.csv, figures/measurement_audit.png.
  D. Does label skew predict per-target exposure beyond client size?  (frontier targets, committed paired scores)

Run from the repository root:
    uv run --with pandas --with scipy --with matplotlib python research/cia_roadmap_v2/scripts/measurement_audit.py
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("REPO_ROOT", os.getcwd())
OUT = os.path.join(ROOT, "research", "cia_roadmap_v2")
W1 = os.path.join(ROOT, "results/cia_frontier/side_channel_audit/results")
W2 = os.path.join(ROOT, "results/cia_frontier/signature_suppression/results/stage_a")
CK = os.path.join(ROOT, "results/cia_frontier/consensus_kl/results")
S3 = os.path.join(CK, "step3")
C = 5.0
FRONTIER_RATIO = 0.001546
WP2_TARGETS = (23, 7, 3, 0, 1)


def training_json(rundir):
    hits = [p for p in glob.glob(f"{rundir}/*.json") if re.search(r"__(eurosat_cnn|paper_cnn)\.json$", p)]
    return json.load(open(hits[0]))


def noise_residual(rundir):
    rows = [dict(round=int(r), signal=a["dp-signal-update-norm"], expected=a["dp-expected-noise-l2-norm"])
            for r, a in training_json(rundir)["train_metrics"].items()]
    df = pd.DataFrame(rows).set_index("round").sort_index()
    rel = pd.DataFrame(json.load(open(f"{rundir}/releases.json"))).set_index("round")
    df = df.join(rel["update_norm"])
    return df.update_norm ** 2 - df.signal ** 2 - df.expected ** 2


def world_of(name):
    return "IN" if "-in-" in name else int(re.search(r"-out-(\d+)-", name).group(1))


def check_a():
    rows = []
    recs = []
    for p in glob.glob(f"{W1}/**/releases.json", recursive=True):
        d = os.path.dirname(p)
        dataset, clients, mech = os.path.relpath(d, W1).split("/")[:3]
        recs.append(dict(dataset=dataset, clients=clients, mech=mech, seed=int(re.search(r"seed-(\d+)", d).group(1)),
                         world=world_of(os.path.basename(d)), dir=d))
    R = pd.DataFrame(recs)
    for (dataset, clients, mech), g in R.groupby(["dataset", "clients", "mech"]):
        res = {(r.seed, r.world): noise_residual(r.dir) for r in g.itertuples()}
        seeds = sorted(g.seed.unique())
        for s in seeds:
            for (s2, w), x in res.items():
                if s2 == s and w != "IN":
                    rows.append(dict(source="WP1 3-8 clients", mech=mech, cell=f"{dataset} {clients}",
                                     pair="same seed, IN vs OUT", corr=float(np.corrcoef(res[(s, "IN")], x)[0, 1])))
            for s2 in seeds:
                if s2 > s:
                    rows.append(dict(source="WP1 3-8 clients", mech=mech, cell=f"{dataset} {clients}",
                                     pair="different seeds, IN vs IN",
                                     corr=float(np.corrcoef(res[(s, "IN")], res[(s2, "IN")])[0, 1])))
    for cfg in sorted(os.listdir(S3)):
        dirs = sorted(glob.glob(f"{S3}/{cfg}/*/"))
        res = {world_of(os.path.basename(d.rstrip("/"))): noise_residual(d) for d in dirs}
        for w, x in res.items():
            if w != "IN":
                rows.append(dict(source="step 3, 48 clients", mech="global-dp", cell=cfg, pair="same seed, IN vs OUT",
                                 corr=float(np.corrcoef(res["IN"], x)[0, 1])))
    A = pd.DataFrame(rows)
    A.to_csv(f"{OUT}/tables/noise_coupling_correlations.csv", index=False)
    return A


def measurements(rundir):
    return pd.DataFrame(json.load(open(f"{rundir}/measurements.json")))


def check_b():
    rows = []
    for cfg, ratio in (("gdp-r0.001546", 0.001546), ("gdp-r0.003092", 0.003092), ("gdp-r0.006184", 0.006184)):
        dirs = sorted(glob.glob(f"{S3}/{cfg}/*/"))
        mi_all = measurements([d for d in dirs if "-in-" in d][0])
        for d in dirs:
            w = world_of(os.path.basename(d.rstrip("/")))
            if w == "IN":
                continue
            mo_all = measurements(d)
            mo = mo_all[mo_all.target == w].set_index("round").sort_index()
            mi = mi_all[mi_all.target == w].set_index("round").sort_index()
            g = (mo.clean_loss - mi.clean_loss).values
            sd = np.sqrt(2) * ratio * C * 0.5 * (mo.clean_grad_norm_start + mi.clean_grad_norm_start).values
            rows.append(dict(config=cfg, noise_multiple=round(ratio / FRONTIER_RATIO), target=w,
                             paired_score=float((g > 0).mean()), decoupled_bound=float(np.mean(norm.cdf(g / sd))),
                             median_gap=float(np.median(g)), median_one_round_noise_sd=float(np.median(sd))))
    B = pd.DataFrame(rows)
    B.to_csv(f"{OUT}/tables/decoupled_score_bound.csv", index=False)
    return B


def check_c():
    recs = []
    paths = glob.glob(f"{W2}/**/measurements.json", recursive=True) + glob.glob(f"{CK}/seeds43-45/**/measurements.json", recursive=True)
    for p in paths:
        d = os.path.dirname(p)
        name = os.path.basename(d)
        cfg = os.path.relpath(d, W2).split("/")[0] if d.startswith(W2) else "consensus-kl-0.3"
        recs.append(dict(cfg=cfg, seed=int(re.search(r"seed-(\d+)", name).group(1)), world=world_of(name), dir=d))
    R = pd.DataFrame(recs)
    rows = []
    for cfg in ("vanilla", "weight-decay-5e-4", "fixed-steps", "local-epochs-1", "consensus-kl-0.3"):
        g = R[R.cfg == cfg]
        seeds = sorted(g.seed.unique())
        L = {}
        for r in g.itertuples():
            m = measurements(r.dir)
            for t in WP2_TARGETS:
                if r.world == "IN" or r.world == t:
                    L[(r.seed, "IN" if r.world == "IN" else "OUT", t)] = m[m.target == t].set_index("round").sort_index().clean_loss.values
        for t in WP2_TARGETS:
            for a in seeds:
                for b in seeds:
                    gab = L[(b, "OUT", t)] - L[(a, "IN", t)]
                    rows.append(dict(cfg=cfg, target=t, seed_in=a, seed_out=b, same_seed=a == b, score=float((gab > 0).mean())))
    X = pd.DataFrame(rows)
    X.to_csv(f"{OUT}/tables/cross_seed_scores_wp2.csv", index=False)
    return X


def check_d():
    """Does label skew predict per-target exposure beyond size?  Frontier targets (paired scores, as committed)."""
    from scipy.stats import spearmanr
    FS = pd.read_csv(os.path.join(ROOT, "research/cia_review_2026-10/data/eurosat_frontier_scores.csv"))
    FR = os.path.join(ROOT, "results/cia_frontier/eurosat_frontier/results/frontier")
    rows = []
    for seed in sorted(FS.seed.unique()):
        part = json.load(open(glob.glob(f"{FR}/global-dp-r0.001546-seed-{seed}-in/*/partitions.json")[0]))
        counts = np.array([c["class_counts"] for c in part["clients"]], float)
        total = counts.sum(0)
        prior = total / total.sum()
        for r in FS[FS.seed == seed].itertuples():
            c = counts[[x["target"] for x in part["clients"]].index(r.target)]
            q = c / c.sum()
            nz = q > 0
            rows.append(dict(mechanism=r.mechanism, seed=seed, target=r.target, score=r.score_clean, records=c.sum(),
                             label_kl=float(np.sum(q[nz] * np.log(q[nz] / prior[nz]))),
                             label_share=float(np.sum(q * c / total))))
    L = pd.DataFrame(rows)
    out = []
    for mech, g in L.groupby("mechanism"):
        for f in ("records", "label_kl", "label_share"):
            rho, pv = spearmanr(g[f], g.score)
            out.append(dict(mechanism=mech, feature=f, n=len(g), spearman=rho, p=pv))
    D = pd.DataFrame(out)
    D.to_csv(f"{OUT}/tables/label_skew_vs_exposure.csv", index=False)
    return D


def figure(A, B, X):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "legend.fontsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300,
                         "axes.titlelocation": "left"})
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw=dict(wspace=0.85, width_ratios=[1, 1.05, 0.95]))
    rng = np.random.default_rng(0)
    ax = axes[0]
    groups = [("other\nseed", A[A.pair == "different seeds, IN vs IN"]["corr"], "0.45"),
              ("same\nseed\n3-8 cl.", A[(A.pair == "same seed, IN vs OUT") & (A.source == "WP1 3-8 clients")]["corr"], "#C44E52"),
              ("same\nseed\n48 cl.", A[A.source == "step 3, 48 clients"]["corr"], "#C44E52")]
    for i, (lab, v, col) in enumerate(groups):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=6, color=col, alpha=0.5, lw=0)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color="k", lw=1.2)
        ax.text(i, 1.1, f"n={len(v)}", ha="center", fontsize=6)
    ax.set_xticks(range(3), [g[0] for g in groups])
    ax.set_ylim(-0.45, 1.18)
    ax.axhline(0, color="0.8", lw=0.6, zorder=0)
    ax.set_ylabel("correlation of realized noise energy\nbetween two trajectories")
    ax.set_title("a  IN and OUT share noise draws")
    ax = axes[1]
    s = B.groupby("noise_multiple")[["paired_score", "decoupled_bound"]].mean()
    ax.plot(s.index, s.paired_score, "o-", color="#C44E52", label="shared noise (as measured)")
    ax.plot(s.index, s.decoupled_bound, "s--", color="#4C72B0", label="independent noise,\nfirst-order estimate (one round)")
    ax.axhline(0.5, color="0.6", lw=0.8, ls=":")
    ax.text(4.6, 0.51, "no signal", fontsize=6, color="0.4", ha="right")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4], ["1x", "2x", "4x"])
    ax.minorticks_off()
    ax.set_xlim(0.85, 4.7)
    ax.set_ylim(0.45, 0.92)
    ax.set_xlabel("global-DP noise (multiple of frontier)")
    ax.set_ylabel("clean-view score, 48 clients\n(5 targets, seed 42)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=1)
    ax.set_title("b  Shared noise hides noise's effect")
    ax = axes[2]
    order = ["vanilla", "weight-decay-5e-4", "fixed-steps", "consensus-kl-0.3", "local-epochs-1"]
    labels = ["vanilla", "weight decay", "fixed steps", "consensus KL", "1 local epoch"]
    for i, cfg in enumerate(order):
        g = X[X.cfg == cfg]
        a, b = g[g.same_seed].score.mean(), g[~g.same_seed].score.mean()
        ax.plot([a, b], [i, i], color="0.75", lw=1.5, zorder=1)
        ax.scatter([a], [i], color="#C44E52", s=18, zorder=2, label="same seed (as measured)" if i == 0 else None)
        ax.scatter([b], [i], color="#4C72B0", s=18, marker="s", zorder=2, label="different seeds" if i == 0 else None)
    ax.set_yticks(range(len(order)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0.6, 0.92)
    ax.set_xlabel("clean-view score, no DP\n(5 targets x 4 seeds)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=1)
    ax.set_title("c  Shared training seeds add more")
    fig.savefig(f"{OUT}/figures/measurement_audit.png", bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(f"{OUT}/tables", exist_ok=True)
    os.makedirs(f"{OUT}/figures", exist_ok=True)
    A, B, X = check_a(), check_b(), check_c()
    Dk = check_d()
    figure(A, B, X)
    pd.set_option("display.width", 200)
    print(A.groupby(["source", "pair"])["corr"].describe()[["count", "mean", "min", "50%", "max"]].round(3).to_string())
    print(B.groupby("noise_multiple")[["paired_score", "decoupled_bound", "median_gap", "median_one_round_noise_sd"]].mean().round(3).to_string())
    print(X.groupby(["cfg", "same_seed"]).score.mean().unstack().round(3).to_string())
    print(Dk.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
