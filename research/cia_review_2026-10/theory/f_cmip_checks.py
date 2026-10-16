"""Numerical checks for f_cmip.md (WP3, D3). No training; committed logs only for the regime evaluation.

A. Proposition 1-2 (crowd bound, k-dimensional statistic): Monte Carlo trade-off against G_mu.
B. Proposition 3 (rounds): composed parameter under independent, adversarially correlated and constant crowd terms.
C. Proposition 4 (shadow sample of size m): exact one-dimensional trade-off with unequal variances (illustrative).
D. Proposition 5 (participation gap): first-order formula against the exact loss gap on a quadratic toy problem,
   for renormalized and fixed-denominator aggregation.
E. Regimes (gate G3, D3 part): rho_r^2 = sigma^2 g_r^2 / V_r from the shadow-loss increments of committed runs
   (WP1 global-DP cells, seeds 42-45; WP2 vanilla 48-client trajectories, seeds 42-45, frontier noise added
   analytically), composed ratio sqrt(mean_r rho_r^2), rule fixed in STATE.md before computing.

    uv run --no-sync --with matplotlib python research/cia_review_2026-10/theory/f_cmip_checks.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from collections import defaultdict
from statistics import NormalDist

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PHI, PHI_INV = NormalDist().cdf, NormalDist().inv_cdf
SEED = 1729
ALPHAS = (0.01, 0.05, 0.1, 0.25)


def write_csv(name, rows):
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    path = os.path.join(HERE, "data", name)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def g_mu(alpha, mu):
    return PHI(PHI_INV(1 - alpha) - mu)


def empirical_type2(stat_in, stat_out, alpha):
    return float((stat_in <= np.quantile(stat_out, 1 - alpha)).mean())


# ---------------------------------------------------------------- A
def check_crowd_bound(samples=300_000, d=6, n=8, C=1.0, sigma=0.05):
    rng = np.random.default_rng(SEED)
    root = rng.normal(size=(d, d))
    crowd = root @ root.T * 0.02  # covariance of the sum of the other clients' updates
    target = rng.normal(size=d)
    target *= C / np.linalg.norm(target)
    S = sigma ** 2 * np.eye(d) + crowd / n ** 2
    a = target / n
    mu_c = float(np.sqrt(a @ np.linalg.solve(S, a)))
    mu_gdp = C / (n * sigma)
    w = rng.normal(size=d)
    w *= np.sign(w @ a)  # orient the statistic so that larger values mean IN
    mu_1 = float(abs(w @ a) / np.sqrt(w @ S @ w))
    L = np.linalg.cholesky(S)
    rows = []
    for name, mu, direction in (("full release (Prop 1)", mu_c, np.linalg.solve(S, a)), ("k = 1 statistic (Prop 2)", mu_1, w)):
        z_in = (rng.normal(size=(samples, d)) @ L.T + a) @ direction
        z_out = (rng.normal(size=(samples, d)) @ L.T) @ direction
        for alpha in ALPHAS:
            beta = empirical_type2(z_in, z_out, alpha)
            rows.append(dict(case=name, mu=round(mu, 4), mu_gdp=round(mu_gdp, 4), ratio_to_gdp=round(mu / mu_gdp, 4),
                             alpha=alpha, empirical_type2=round(beta, 4), g_mu_type2=round(g_mu(alpha, mu), 4),
                             difference=round(beta - g_mu(alpha, mu), 4)))
    return write_csv("f_cmip_crowd_bound.csv", rows), rows


# ---------------------------------------------------------------- B
def temporal_cases(T=50):
    rng = np.random.default_rng(SEED)
    s = np.full(T, 1.0)  # per-round signal (worst case, equal)
    D = np.full(T, 1.0)  # DP-noise variance of the statistic
    tau2 = np.full(T, 1.0)  # crowd variance per round
    mu = lambda gamma: float(np.sqrt(s @ np.linalg.solve(np.diag(D) + gamma, s)))
    independent = mu(np.diag(tau2))
    constant = mu(np.outer(np.sqrt(tau2), np.sqrt(tau2)))
    # Adversarial: rank-one crowd with the given per-round variances whose pattern is D^{-1}-orthogonal to s.
    signs = np.where(np.arange(T) % 2 == 0, 1.0, -1.0)
    v = signs * np.sqrt(tau2)
    adversarial = mu(np.outer(v, v))
    gdp = float(np.sqrt(s @ (s / D)))
    rows = [dict(case=c, T=T, mu=round(m, 4), ratio_to_gdp=round(m / gdp, 4)) for c, m in (
        ("GDP (crowd known)", gdp), ("independent crowd across rounds", independent),
        ("constant crowd across rounds", constant), ("adversarial rank-one crowd (sums to zero along s)", adversarial))]
    del rng
    return write_csv("f_cmip_temporal.csv", rows), rows


# ---------------------------------------------------------------- C
def binormal_type2(a, s0, s1, alpha, grid=200_001):
    """Neyman-Pearson type II error, OUT = N(0, s0^2) vs IN = N(a, s1^2), on a fine grid."""
    lo, hi = min(-10 * s0, a - 10 * s1), max(10 * s0, a + 10 * s1)
    x = np.linspace(lo, hi, grid)
    p0 = np.exp(-x ** 2 / (2 * s0 ** 2)) / s0
    p1 = np.exp(-(x - a) ** 2 / (2 * s1 ** 2)) / s1
    p0, p1 = p0 / p0.sum(), p1 / p1.sum()
    order = np.argsort(-(np.log(p1 + 1e-300) - np.log(p0 + 1e-300)))
    fpr, tpr = np.cumsum(p0[order]), np.cumsum(p1[order])
    return float(1 - np.interp(alpha, fpr, tpr))


def partial_knowledge(C=1.0, n=8, sigma=0.05, tau_ratio=1.0, idio_ratio=0.5):
    """Illustrative model (assumptions in f_cmip.md, Prop 4): alignment kappa(m) = 1/sqrt(1 + 32/m)."""
    rows = []
    s_noise = sigma
    s0 = math.hypot(s_noise, tau_ratio * sigma)
    for m in (1, 8, 32, 128, 512, 10 ** 6):
        kappa = 1 / math.sqrt(1 + 32 / m)
        a = kappa * C / n
        s1 = math.hypot(s0, idio_ratio * C / n)
        beta = binormal_type2(a, s0, s1, 0.05)
        mu_equivalent = PHI_INV(0.95) - PHI_INV(beta)  # G_mu with the same type II error at alpha = 0.05
        rows.append(dict(shadow_size=m, alignment=round(kappa, 4), type2_at_5pct=round(beta, 4),
                         mu_equivalent_at_5pct=round(mu_equivalent, 4),
                         crowd_bound_mu=round((C / n) / s0, 4), gdp_mu=round((C / n) / s_noise, 4)))
    return write_csv("f_cmip_partial_knowledge.csv", rows), rows


# ---------------------------------------------------------------- D
def participation_gap(n=8, d=20, K=5):
    rng = np.random.default_rng(SEED)
    A = [(lambda M: M @ M.T / d + 0.1 * np.eye(d))(rng.normal(size=(d, d))) for _ in range(n)]
    b = [rng.normal(size=d) + 2 * rng.normal(size=d) * (j == 0) for j in range(n)]
    loss = lambda j, th: 0.5 * th @ A[j] @ th - b[j] @ th
    grad = lambda j, th: A[j] @ th - b[j]
    theta = rng.normal(size=d)
    rows = []
    for eta in (0.1, 0.03, 0.01, 0.003):
        def local(j):
            th = theta.copy()
            for _ in range(K):
                th -= eta * grad(j, th)
            return th - theta
        updates = [local(j) for j in range(n)]
        target, others = 0, list(range(1, n))
        g = [grad(j, theta) for j in range(n)]
        for convention in ("renormalized", "fixed denominator"):
            if convention == "renormalized":
                delta_in = np.mean(updates, axis=0)
                delta_out = np.mean([updates[j] for j in others], axis=0)
                gbar = np.mean([g[j] for j in others], axis=0)
                first_order = (1 / n) * eta * K * g[target] @ (g[target] - gbar)
            else:
                delta_in = np.sum(updates, axis=0) / n
                delta_out = np.sum([updates[j] for j in others], axis=0) / n
                first_order = (1 / n) * eta * K * g[target] @ g[target]
            exact = loss(target, theta + delta_out) - loss(target, theta + delta_in)
            rows.append(dict(convention=convention, eta=eta, K=K, eta_K=round(eta * K, 4), exact_gap=exact,
                             first_order=first_order, relative_error=abs(first_order - exact) / abs(exact)))
    return write_csv("f_cmip_participation_gap.csv", rows), rows


# ---------------------------------------------------------------- E
def out_increments(folder_glob, out_filter=None):
    """{target: {seed: (increments, grad_norms)}} over the OUT trajectories matching the glob."""
    data = defaultdict(dict)
    for manifest_path in glob.glob(folder_glob):
        manifest = json.load(open(manifest_path))
        target = manifest["out_target"]
        if target is None or (out_filter and not out_filter(manifest)):
            continue
        rows = [r for r in json.load(open(manifest_path.replace("manifest.json", "measurements.json")))
                if r["target"] == target]
        rows.sort(key=lambda r: r["round"])
        data[target][manifest["seed"]] = (np.array([r["clean_increment"] for r in rows]),
                                          np.array([r["clean_grad_norm_start"] for r in rows]))
    return data


def regime_ratios(data, sigma, noise_in_runs):
    """Per round rho_r^2 = mean(sigma^2 g^2) / V_r with V_r pooled over targets around each target's seed mean.

    ``noise_in_runs``: the trajectories already contain the DP noise (V_r includes it); otherwise the noise term is
    added analytically, V_r = crowd variance + mean(sigma^2 g^2).
    """
    squares, dof, noise, count = 0.0, 0, 0.0, 0
    for seeds in data.values():
        if len(seeds) < 2:
            continue
        increments = np.array([v[0] for v in seeds.values()])
        grads = np.array([v[1] for v in seeds.values()])
        squares = squares + ((increments - increments.mean(axis=0)) ** 2).sum(axis=0)
        dof += len(seeds) - 1
        noise = noise + (sigma ** 2 * grads ** 2).sum(axis=0)
        count += len(seeds)
    variance = squares / dof
    noise = noise / count
    total = variance if noise_in_runs else variance + noise
    rho2 = np.minimum(noise / total, 1.0)
    return rho2, dof


def meaningful(rho2, sigma, clients, rounds, C=5.0, mu_total=1.0):
    """Extrapolation (crowd variance held fixed): noise multiple for a mu_total-GDP guarantee and the ratio there.

    Per round GDP mu = (C / n) / sigma'; mu_total = sqrt(T) mu. With c_r = (1 - rho_r^2) / rho_r^2 the crowd-to-noise
    variance ratio at sigma, rho_r(sigma') = 1 / sqrt(1 + c_r (sigma / sigma')^2).
    """
    target = (C / clients) / (mu_total / math.sqrt(rounds))
    multiple = target / sigma
    crowd = (1 - rho2) / np.maximum(rho2, 1e-12)
    rho_new = 1 / np.sqrt(1 + crowd / multiple ** 2)
    return multiple, float(np.sqrt((rho_new ** 2).mean()))


def regimes():
    rows, curves = [], {}
    audit = os.path.join(ROOT, "results/cia_frontier/side_channel_audit/results")
    for dataset, ratio in (("eurosat", 0.001546), ("alzheimer", 0.01 / 3)):
        for clients in (3, 5, 8):
            data = out_increments(f"{audit}/{dataset}/clients-{clients}/global-dp/*/manifest.json")
            if not data:
                continue
            sigma = ratio * 5.0
            rho2, dof = regime_ratios(data, sigma, noise_in_runs=True)
            label = f"WP1 {dataset} {clients} clients, global DP sigma={sigma:.5f} (in the runs)"
            curves[label] = np.sqrt(rho2)
            runnable = dataset == "eurosat" and clients == 8
            rows.append(dict(regime=label, dataset=dataset, clients=clients, rounds=len(rho2), targets=len(data),
                             pooled_dof=dof, sigma=round(sigma, 6), median_rho=round(float(np.median(np.sqrt(rho2))), 4),
                             composed_ratio=round(float(np.sqrt(rho2.mean())), 4), runnable_for_gate=runnable,
                             tighter_by_20pct=bool(np.sqrt(rho2.mean()) <= 0.8),
                             gdp_mu_total=round((5.0 / clients) / sigma * math.sqrt(len(rho2)), 1),
                             **dict(zip(("noise_multiple_for_mu1", "ratio_at_mu1_extrapolated"),
                                        (round(x, 4) for x in meaningful(rho2, sigma, clients, len(rho2)))))))
    stage_a = os.path.join(ROOT, "results/cia_frontier/signature_suppression/results/stage_a/vanilla")
    data = out_increments(f"{stage_a}/*/manifest.json")
    for multiple in (1, 2, 4):
        sigma = 0.001546 * 5.0 * multiple
        rho2, dof = regime_ratios(data, sigma, noise_in_runs=False)
        label = f"WP2 eurosat 48 clients, vanilla runs + sigma={sigma:.5f} added ({multiple}x frontier)"
        curves[label] = np.sqrt(rho2)
        rows.append(dict(regime=label, dataset="eurosat", clients=48, rounds=len(rho2), targets=len(data),
                         pooled_dof=dof, sigma=round(sigma, 6), median_rho=round(float(np.median(np.sqrt(rho2))), 4),
                         composed_ratio=round(float(np.sqrt(rho2.mean())), 4), runnable_for_gate=multiple == 1,
                         tighter_by_20pct=bool(np.sqrt(rho2.mean()) <= 0.8),
                         gdp_mu_total=round((5.0 / 48) / sigma * math.sqrt(len(rho2)), 1),
                         **dict(zip(("noise_multiple_for_mu1", "ratio_at_mu1_extrapolated"),
                                    (round(x, 4) for x in meaningful(rho2, sigma, 48, len(rho2)))))))
    return write_csv("f_cmip_regimes.csv", rows), rows, curves


def figure(curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for label, curve in curves.items():
        if "eurosat" not in label or ("WP2" in label and "1x" not in label):
            continue
        ax.plot(np.arange(1, len(curve) + 1), curve, lw=1.4, label=label.split(", global")[0].split(", vanilla")[0])
    ax.axhline(0.8, color="#c0392b", ls="--", lw=1, label="20% tighter than GDP")
    ax.axhline(1.0, color="#555", lw=0.8)
    ax.set_xlabel("round")
    ax.set_ylabel("rho_r = per-round bound / GDP bound")
    ax.set_ylim(0, 1.05)
    ax.set_title("Loss-statistic attacker: crowd bound relative to GDP, EuroSAT", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    path = os.path.join(HERE, "figures", "f_cmip_regimes.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    for name, result in (("A", check_crowd_bound()), ("B", temporal_cases()), ("C", partial_knowledge()),
                         ("D", participation_gap())):
        print(name, result[0])
    path, rows, curves = regimes()
    print("E", path)
    print(figure(curves))


if __name__ == "__main__":
    main()
