"""Where can Gaussian membership-inference privacy (GMIP) buy anything at client level?

Pure arithmetic on the formulas printed in Leemann, Pawelczyk & Kasneci, "Gaussian Membership
Inference Privacy", NeurIPS 2023 (papers/Gaussian Membership Inference Privacy.pdf):

  * Corollary 5.1 (p. 7):  mu_step = (d + (2 n_eff - 1) K) / (n_eff * sqrt(2 d + 4 n_eff K)),
                           n_eff = n + tau^2 n^2 / C^2    (Theorem 5.1)
  * mu-GDP of one Gaussian release of the mean of n clipped vectors under replacement adjacency:
                           mu_GDP = 2 C / (n tau)          (Dong, Roth & Su 2022)

Transfer to FL (our reading, to be proven, NOT a result of the paper): the "sample gradient" becomes
a whole client's clipped update, n becomes the number of clients aggregated per round, d the number
of shared parameters (or the dimension of the statistic the attacker can use), and the background
randomness becomes the between-client variability of updates. K = d corresponds to a typical
client (Remark 5.1); outlier clients have K >> d.

Run:  uv run --with scipy --with pandas --with matplotlib python research/cia_review_2026-10/scripts/gmip_regime_calculator.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("REPO_ROOT", os.getcwd())
OUT = os.path.join(ROOT, "research", "cia_review_2026-10")


def mu_step(d, n, tau=0.0, C=1.0, K=None):
    K = d if K is None else K
    ne = n + (tau ** 2) * n ** 2 / C ** 2
    return (d + (2 * ne - 1) * K) / (ne * np.sqrt(2 * d + 4 * ne * K))


def tau_for_mu_gmip(d, n, mu, C=1.0, K=None):
    """Smallest tau with mu_step <= mu (bisection on n_eff; mu_step decreases in n_eff)."""
    if mu_step(d, n, 0.0, C, K) <= mu:
        return 0.0
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mu_step(d, n, mid, C, K) <= mu:
            hi = mid
        else:
            lo = mid
    return hi


def tau_for_mu_gdp(n, mu, C=1.0):
    return 2 * C / (n * mu)


REGIMES = [
    # name, d (shared dimension / attacker statistic dimension), n (vectors averaged per release)
    ("GMIP paper: CIFAR-10 last layer, record level, batch 1000", 650, 1000),
    ("This repo: EuroSAT CNN full model, 48 clients", 289_194, 48),
    ("This repo: stacked head, 132 head parameters, 8 clients", 132, 8),
    ("This repo: stacked head projected, 51 dims, 8 clients", 51, 8),
    ("Cross-device LoRA-style adapter, 10k params, 1000 clients/round", 10_000, 1000),
    ("Scalar loss statistic (shadow-loss attacker), 48 clients", 1, 48),
]


def main():
    rows = []
    for name, d, n in REGIMES:
        r = dict(regime=name, d=d, n=n, mu_step_noiseless=float(mu_step(d, n)))
        for mu in (0.5, 1.0, 2.0):
            tg = tau_for_mu_gmip(d, n, mu)
            td = tau_for_mu_gdp(n, mu)
            r[f"tau_gmip_mu{mu}"] = tg
            r[f"tau_gdp_mu{mu}"] = td
            r[f"gmip_over_gdp_noise_mu{mu}"] = tg / td
        # outlier client: K = 10 d
        r["mu_step_noiseless_outlier_K10d"] = float(mu_step(d, n, K=10 * d))
        rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(OUT, "data"), exist_ok=True)
    df.to_csv(os.path.join(OUT, "data/gmip_regimes.csv"), index=False)

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6,
                         "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 300,
                         "axes.titlelocation": "left"})
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ns = np.logspace(0.5, 4.5, 200)
    cols = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
    for d, c in zip((1, 51, 650, 10_000, 289_194), cols):
        axs[0].plot(ns, mu_step(d, ns), color=c, lw=1.2, label=f"d = {d:,}")
    axs[0].axhline(1.0, color="0.5", ls=":", lw=0.8)
    axs[0].set_xscale("log"); axs[0].set_yscale("log")
    axs[0].set_xlabel("vectors averaged per release, n (clients per round)")
    axs[0].set_ylabel("noiseless one-step GMIP parameter mu_step")
    axs[0].set_title("Crowd privacy needs n comparable to d")
    axs[0].legend(loc="lower left")
    ax = axs[1]
    labels = []
    for k, r in df.iterrows():
        v = r["gmip_over_gdp_noise_mu1.0"]
        ax.scatter(max(v, 1e-3), k, color="#DD8452" if v > 1 else "#55A868", s=18, zorder=3)
        ax.hlines(k, 1e-3, max(v, 1e-3), color="0.8", lw=1, zorder=1)
        labels.append(r.regime.replace(", ", "\n", 1))
    ax.axvline(1.0, color="0.4", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(range(len(df)), labels, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel("noise std for mu = 1, GMIP bound / GDP")
    ax.text(0.98, 0.02, "left of 1 = GMIP cheaper\n1e-3 = no noise needed", transform=ax.transAxes, ha="right", va="bottom", fontsize=6)
    ax.set_title("GMIP's bound only helps when n >= d")
    for a, l in zip(axs, "ab"):
        a.text(-0.16 if l == "a" else -0.9, 1.05, l, transform=a.transAxes, fontweight="bold", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "figures/gmip_regimes.png"))
    plt.close(fig)
    return df


if __name__ == "__main__":
    print(main().round(4).to_string())
