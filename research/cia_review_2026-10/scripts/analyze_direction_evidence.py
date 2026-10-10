"""Evidence for research_directions.md, computed from committed EuroSAT frontier logs only (no training).

Inputs: results/cia_frontier/eurosat_frontier/results/frontier/<run>/<run-dir>/*eurosat_cnn.json (training
metrics, 48 clients, 100 rounds, alpha .3, noise ratio .001546, seeds 42-44, targets 0-9) and
research/cia_review_2026-10/data/eurosat_frontier_scores.csv (per-target clean/noisy scores, written by
analyze_project_history.py).

Outputs (research/cia_review_2026-10/):
  data/noise_level_side_channel_48client.csv    E1  IN/OUT noise-level comparison per target
  data/dispersion_statistic_loo_sensitivity.csv E3  leave-one-client-out shift of max/q90/median/mean distance
  data/target_idiosyncrasy_vs_cia_score.csv     E4  per-target weight, relative distance, update norm, scores
  data/target_idiosyncrasy_correlations.csv     E4  Spearman correlations
  data/target_idiosyncrasy_partial.csv          E4  collinearity and partial rank correlation given weight
  data/oracle_projection_snr_proxy.csv          E6  w_i * min(||u_i||, C) / sigma per target (median over rounds)
  figures/direction_evidence.png

Run from the repository root:
    uv run --with pandas --with scipy --with matplotlib python research/cia_review_2026-10/scripts/analyze_direction_evidence.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("REPO_ROOT", os.getcwd())
FR = os.path.join(ROOT, "results/cia_frontier/eurosat_frontier/results/frontier")
OUT = os.path.join(ROOT, "research/cia_review_2026-10")
D = 289_194  # EurosatCNN float parameters
C = 5.0
SEEDS = (42, 43, 44)
MECHS = ("global-dp", "metric-privacy")
RATIO = "r0.001546"


def load_train(run):
    ps = [p for p in glob.glob(f"{FR}/{run}/*/*.json") if p.endswith("eurosat_cnn.json")]
    return json.load(open(ps[0])) if ps else None


def noise_series(run):
    rows = []
    for r, a in load_train(run)["train_metrics"].items():
        rows.append(dict(round=int(r), noise=a["dp-expected-noise-l2-norm"], signal=a["dp-signal-update-norm"], n=len(a["client-ids"])))
    return pd.DataFrame(rows).set_index("round").sort_index()


def e1_side_channel():
    out = []
    for mech in MECHS:
        for seed in SEEDS:
            IN = noise_series(f"{mech}-{RATIO}-seed-{seed}-in")
            for k in range(10):
                run = f"{mech}-{RATIO}-seed-{seed}-out-{k}"
                if not os.path.isdir(f"{FR}/{run}"):
                    continue
                j = IN.join(noise_series(run), lsuffix="_in", rsuffix="_out", how="inner")
                # what a blind observer of ||theta_{t+1}-theta_t|| sees in expectation: sqrt(noise^2 + signal^2)
                obs_in = np.sqrt(j.noise_in ** 2 + j.signal_in ** 2)
                obs_out = np.sqrt(j.noise_out ** 2 + j.signal_out ** 2)
                out.append(dict(mechanism=mech, seed=seed, target=k, rounds=len(j), n_in=int(j.n_in.iloc[0]), n_out=int(j.n_out.iloc[0]),
                                noise_to_signal_in_median=float((j.noise_in / j.signal_in).median()),
                                true_noise_logratio_mean=float(np.log(j.noise_in / j.noise_out).mean()),
                                observed_norm_logratio_mean=float(np.log(obs_in / obs_out).mean()),
                                frac_rounds_observed_in_lt_out=float((obs_in < obs_out).mean())))
    df = pd.DataFrame(out)
    df.to_csv(f"{OUT}/data/noise_level_side_channel_48client.csv", index=False)
    return df


def pairwise(a):
    return (np.asarray(a["metric-dp-pairwise-distances"], float), np.asarray(a["metric-dp-pairwise-client-i"]),
            np.asarray(a["metric-dp-pairwise-client-j"]))


def e3_loo():
    rows = []
    stats = dict(max=np.max, q90=lambda x: np.quantile(x, 0.9), median=np.median, mean=np.mean)
    for seed in SEEDS:
        for r, a in load_train(f"metric-privacy-{RATIO}-seed-{seed}-in")["train_metrics"].items():
            d, ii, jj = pairwise(a)
            full = {k: f(d) for k, f in stats.items()}
            for x in np.unique(np.concatenate([ii, jj])):
                keep = (ii != x) & (jj != x)
                rows.append(dict(seed=seed, round=int(r), client=int(x), **{f"shift_{k}": float(np.log(full[k] / f(d[keep]))) for k, f in stats.items()}))
    L = pd.DataFrame(rows)
    summ = {}
    for s in stats:
        col = f"shift_{s}"
        per_client = L.groupby(["seed", "client"])[col].mean().abs()
        summ[s] = dict(max_client_mean_abs_shift_pct=100 * per_client.max(), p99_round_abs_shift_pct=100 * L[col].abs().quantile(0.99),
                       max_round_abs_shift_pct=100 * L[col].abs().max())
    S = pd.DataFrame(summ).T
    S.index.name = "statistic"
    S.to_csv(f"{OUT}/data/dispersion_statistic_loo_sensitivity.csv")
    return S


def client_features(run):
    acc = {}
    for r, a in load_train(run)["train_metrics"].items():
        d, ii, jj = pairwise(a)
        ids = np.asarray(a["client-ids"])
        ne = np.asarray(a["per-client-num-examples"], float)
        w = dict(zip(ids, ne / ne.sum()))
        un = dict(zip(np.asarray(a["dp-client-ids"]), np.asarray(a["dp-update-norms-before-clipping"], float)))
        med = np.median(d)
        for x in ids:
            m = (ii == x) | (jj == x)
            acc.setdefault(int(x), []).append((d[m].mean() / med, w[x], un.get(x, np.nan)))
    return {k: np.nanmean(np.array(v), axis=0) for k, v in acc.items()}


def e4_exposure():
    FS = pd.read_csv(f"{OUT}/data/eurosat_frontier_scores.csv")
    rows = []
    for seed in SEEDS:
        # pairwise distances are logged only by the metric-privacy strategy; use them for both mechanisms
        F = client_features(f"metric-privacy-{RATIO}-seed-{seed}-in")
        for mech in MECHS:
            for k in range(10):
                if k in F:
                    rows.append(dict(mechanism=mech, seed=seed, target=k, rel_mean_distance=F[k][0], weight=F[k][1], update_norm=F[k][2]))
    FT = pd.DataFrame(rows).merge(FS, on=["mechanism", "seed", "target"])
    FT.to_csv(f"{OUT}/data/target_idiosyncrasy_vs_cia_score.csv", index=False)
    corr, part = [], []
    for mech, g in FT.groupby("mechanism"):
        for feat in ["rel_mean_distance", "weight", "update_norm"]:
            for sc in ["score_clean", "score_noisy"]:
                rho, p = spearmanr(g[feat], g[sc])
                corr.append(dict(mechanism=mech, feature=feat, score=sc, n=len(g), spearman=rho, p=p))
        rw, rd, rs = (rankdata(g[c]) for c in ["weight", "rel_mean_distance", "score_clean"])
        res = lambda y, x: y - np.polyval(np.polyfit(x, y, 1), x)
        pr, pp = pearsonr(res(rd, rw), res(rs, rw))
        part.append(dict(mechanism=mech, rho_distance_weight=spearmanr(g.weight, g.rel_mean_distance)[0],
                         rho_updatenorm_weight=spearmanr(g.weight, g.update_norm)[0], partial_rho_distance_score_given_weight=pr, p=pp))
    CORR, PART = pd.DataFrame(corr), pd.DataFrame(part)
    CORR.to_csv(f"{OUT}/data/target_idiosyncrasy_correlations.csv", index=False)
    PART.to_csv(f"{OUT}/data/target_idiosyncrasy_partial.csv", index=False)
    return FT, CORR, PART


def e6_snr():
    rows = []
    for mech in MECHS:
        for seed in SEEDS:
            for r, a in load_train(f"{mech}-{RATIO}-seed-{seed}-in")["train_metrics"].items():
                ids = np.asarray(a["client-ids"])
                ne = np.asarray(a["per-client-num-examples"], float)
                un = dict(zip(np.asarray(a["dp-client-ids"]), np.asarray(a["dp-update-norms-before-clipping"], float)))
                sigma = a["dp-expected-noise-l2-norm"] / np.sqrt(D)
                for x, wx in zip(ids, ne / ne.sum()):
                    if x < 10:
                        rows.append(dict(mechanism=mech, seed=seed, round=int(r), target=int(x), sigma=sigma, snr=wx * min(un[x], C) / sigma))
    SN = pd.DataFrame(rows)
    g = SN.groupby(["mechanism", "seed", "target"]).agg(snr_median=("snr", "median"), sigma_median=("sigma", "median")).reset_index()
    g.to_csv(f"{OUT}/data/oracle_projection_snr_proxy.csv", index=False)
    return g


def figure(FT, S):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300,
                         "axes.titlelocation": "left"})
    col = {"global-dp": "#4C72B0", "metric-privacy": "#DD8452"}
    lab = {"global-dp": "global DP", "metric-privacy": "metric privacy"}
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw=dict(wspace=0.45))
    for mech, g in FT.groupby("mechanism"):
        rho = spearmanr(g.weight, g.score_clean)[0]
        a.scatter(100 * g.weight, g.score_clean, s=16, color=col[mech], alpha=0.8, label=f"{lab[mech]} (Spearman {rho:.2f})")
    a.axhline(0.5, color="0.6", lw=0.8, ls="--")
    a.text(0.3, 0.51, "no signal", fontsize=6.5, color="0.4")
    a.set_xlabel("target client's aggregation weight\n(% of examples; EuroSAT, 48 clients; 24 target-seed pairs)")
    a.set_ylabel("per-target clean-view score\n(share of rounds IN loss < OUT loss)")
    a.set_title("a  Larger clients are more exposed")
    a.legend(loc="lower right", fontsize=6.5)
    stats = ["max", "q90", "median", "mean"]
    x = np.arange(len(stats))
    b.bar(x - 0.2, S.loc[stats, "max_client_mean_abs_shift_pct"], width=0.4, color="0.55", label="most exposed client,\nmean over rounds")
    b.bar(x + 0.2, S.loc[stats, "max_round_abs_shift_pct"], width=0.4, color="0.25", label="worst single round,\nany client")
    for xi, s in zip(x, stats):
        b.text(xi + 0.2, S.loc[s, "max_round_abs_shift_pct"] + 0.2, f"{S.loc[s, 'max_round_abs_shift_pct']:.1f}", ha="center", fontsize=6.5)
        b.text(xi - 0.2, S.loc[s, "max_client_mean_abs_shift_pct"] + 0.2, f"{S.loc[s, 'max_client_mean_abs_shift_pct']:.1f}", ha="center", fontsize=6.5)
    b.set_xticks(x, ["max\n(paper)", "90th\npercentile", "median", "mean"])
    b.set_ylabel("|log change| on removing\none client (%)")
    b.set_title("b  Robust dispersion statistics move less")
    b.legend(loc="upper right", fontsize=6.5)
    fig.savefig(f"{OUT}/figures/direction_evidence.png", bbox_inches="tight")
    plt.close(fig)


def main():
    sc = e1_side_channel()
    S = e3_loo()
    FT, CORR, PART = e4_exposure()
    g = e6_snr()
    figure(FT, S)
    pd.set_option("display.width", 160)
    print("E1", sc.groupby("mechanism")[["noise_to_signal_in_median", "true_noise_logratio_mean", "frac_rounds_observed_in_lt_out"]].agg(["mean", "min", "max"]).round(4).to_string())
    print("E3", S.round(2).to_string())
    print("E4", CORR[CORR.score == "score_clean"].round(3).to_string(index=False))
    print(PART.round(3).to_string(index=False))
    print("E6", g.groupby("mechanism")[["snr_median", "sigma_median"]].median().round(4).to_string())


if __name__ == "__main__":
    main()
