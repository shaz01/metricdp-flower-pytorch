"""Numerical checks for private_dispersion.md (WP3, D4 item 1). No training; committed logs only.

1. Monte Carlo check of Lemma 3: the empirical trade-off of one round (bit count + update release, worst-case
   target: bit 1, update norm C, quantile at q_max) against G_{mu_round}.
2. Accounting table: mu_total, (epsilon, delta = 1e-5) and the median share for the default 10% bit noise.
3. Tracking accuracy: the private median q_t against the true median of the per-client statistic, replayed on
   per-client update norms logged by committed runs (EuroSAT frontier, 48 clients, 100 rounds; WP1 EuroSAT,
   8 clients, 50 rounds). The tracker is scale-free in log space, so update norms stand in for the model distance.

    uv run --no-sync --with matplotlib python research/cia_review_2026-10/theory/private_dispersion_checks.py
Writes theory/data/private_dispersion_*.csv and theory/figures/private_dispersion_tracking.png.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
from statistics import NormalDist

import numpy as np

from metricdp_pytorch.private_metric_strategy import gaussian_epsilon, privacy_account, quantile_update

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PHI, PHI_INV = NormalDist().cdf, NormalDist().inv_cdf
SEED = 1729


def write_csv(name, rows):
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    path = os.path.join(HERE, "data", name)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def g_mu(alpha, mu):
    """Type II error of the GDP trade-off function at type I error alpha."""
    return PHI(PHI_INV(1 - alpha) - mu)


def check_one_round(samples=400_000):
    """Lemma 3 with equality case: n = 4 slots, C = 1, z and sigma_b from the 10% rule, q_t = q_max."""
    rng = np.random.default_rng(SEED)
    rows = []
    for mu_round in (0.5, 1.0, 2.0):
        mu_u, mu_b = math.sqrt(0.9) * mu_round, math.sqrt(0.1) * mu_round
        n, C = 4, 1.0
        sigma = (C / n) / mu_u  # update noise std at the floor
        sigma_b = 1 / mu_b
        # Sufficient statistics: the update projected on the target's (norm-C) update, and the bit count.
        def draw(present):
            update = present * C / n + rng.normal(0, sigma, samples)
            bits = present * 1.0 + rng.normal(0, sigma_b, samples)
            return update / sigma ** 2 * (C / n) + bits / sigma_b ** 2  # log-LR up to a constant
        llr_in, llr_out = draw(1), draw(0)
        for alpha in (0.01, 0.05, 0.1, 0.25):
            threshold = np.quantile(llr_out, 1 - alpha)
            beta = float((llr_in <= threshold).mean())
            rows.append(dict(mu_round=mu_round, alpha=alpha, empirical_type2=round(beta, 4),
                             g_mu_type2=round(g_mu(alpha, mu_round), 4),
                             difference=round(beta - g_mu(alpha, mu_round), 4)))
    return write_csv("private_dispersion_one_round.csv", rows), rows


def accounting_table():
    rows = []
    for label, clients, rounds, z, q_max in (
        ("frontier noise (r = 0.001546, z = r n), q_max = C", 48, 100, 0.001546 * 48, 5.0),
        ("target mu_total = 1, 48 clients, 100 rounds", 48, 100, None, 5.0),
        ("target mu_total = 4, 48 clients, 100 rounds", 48, 100, None, 5.0),
        ("target mu_total = 1, 8 clients, 50 rounds", 8, 50, None, 5.0),
        ("target mu_total = 4, 8 clients, 50 rounds", 8, 50, None, 5.0),
    ):
        if z is None:
            mu_total = float(label.split("= ")[1].split(",")[0])
            mu_round = mu_total / math.sqrt(rounds)
            z = q_max / (math.sqrt(0.9) * mu_round)
        sigma_b = 3 * z / q_max
        account = privacy_account(noise_multiplier=z, maximum=q_max, bit_noise_std=sigma_b, rounds=rounds)
        rows.append(dict(setting=label, clients=clients, rounds=rounds, z=round(z, 4), q_max=q_max,
                         sigma_min_on_average=round(z * 5.0 / (clients * q_max), 6), bit_noise_std=round(sigma_b, 4),
                         bit_noise_on_fraction=round(sigma_b / clients, 4),
                         mu_update_per_round=round(account.mu_update_per_round, 4),
                         mu_total=round(account.mu_total, 4), median_share=round(account.median_share, 4),
                         epsilon_at_delta_1e5=round(gaussian_epsilon(account.mu_total, 1e-5), 3)))
    return write_csv("private_dispersion_accounting.csv", rows), rows


def logged_norms():
    """{run label: array (rounds, clients)} of per-client update norms before clipping, IN trajectories."""
    runs = {}
    frontier = os.path.join(ROOT, "results/cia_frontier/eurosat_frontier/results/frontier")
    for mech in ("global-dp", "metric-privacy"):
        for seed in (42, 43, 44):
            files = [p for p in glob.glob(f"{frontier}/{mech}-r0.001546-seed-{seed}-in/*/*.json")
                     if p.endswith("eurosat_cnn.json")]
            if files:
                metrics = json.load(open(files[0]))["train_metrics"]
                runs[f"frontier 48 clients {mech} seed {seed}"] = np.array(
                    [metrics[str(r)]["dp-update-norms-before-clipping"] for r in range(1, len(metrics) + 1)])
    audit = os.path.join(ROOT, "results/cia_frontier/side_channel_audit/results/eurosat/clients-8")
    for mech in ("global-dp", "metric-privacy"):
        for path in sorted(glob.glob(f"{audit}/{mech}/*-in-*/*.json")):
            if path.endswith("eurosat_cnn.json") and "seed-45" not in path:
                metrics = json.load(open(path))["train_metrics"]
                seed = path.split("__seed-")[1].split("__")[0]
                runs[f"WP1 8 clients {mech} seed {seed}"] = np.array(
                    [metrics[str(r)]["dp-update-norms-before-clipping"] for r in range(1, len(metrics) + 1)])
    if not runs:
        raise FileNotFoundError("No committed logs found; add the result folders to the sparse checkout")
    return runs


def track(norms, mu_round, rng, *, q0=5.0, q_min=0.05, q_max=50.0, eta=0.2, gamma=0.5):
    """Replay the tracker on logged statistics; returns |log(q_t / true median_t)| per round."""
    clients = norms.shape[1]
    sigma_b = 1 / (math.sqrt(0.1) * mu_round)
    q, errors = q0, []
    for values in norms:
        errors.append(abs(math.log(q / float(np.median(values)))))
        noisy = (float((values <= q).sum()) + rng.normal(0, sigma_b)) / clients
        q = quantile_update(q, noisy, target=gamma, learning_rate=eta, minimum=q_min, maximum=q_max)
    return np.array(errors)


def tracking(repeats=200):
    rng = np.random.default_rng(SEED)
    rows, curves = [], {}
    for label, norms in logged_norms().items():
        rounds = len(norms)
        for mu_total in (1.0, 4.0, 16.0):
            mu_round = mu_total / math.sqrt(rounds)
            errors = np.array([track(norms, mu_round, rng) for _ in range(repeats)])
            late = errors[:, 10:]
            rows.append(dict(run=label, clients=norms.shape[1], rounds=rounds, mu_total=mu_total,
                             bit_noise_on_fraction=round(1 / (math.sqrt(0.1) * mu_round) / norms.shape[1], 3),
                             median_abs_log_error_rounds_11_on=round(float(np.median(late)), 4),
                             p90_abs_log_error_rounds_11_on=round(float(np.quantile(late, 0.9)), 4),
                             true_median_first=round(float(np.median(norms[0])), 3),
                             true_median_last=round(float(np.median(norms[-1])), 3)))
            curves[(label, mu_total)] = np.median(errors, axis=0)
    return write_csv("private_dispersion_tracking.csv", rows), rows, curves


def figure(curves):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for axis, prefix in zip(axes, ("frontier 48 clients global-dp seed 42", "WP1 8 clients global-dp seed 42")):
        for mu_total, color in zip((1.0, 4.0, 16.0), ("#c0392b", "#e67e22", "#2e86c1")):
            if (prefix, mu_total) in curves:
                curve = curves[(prefix, mu_total)]
                axis.plot(np.arange(1, len(curve) + 1), np.exp(curve) - 1, color=color, label=f"mu_total = {mu_total:g}")
        axis.set_title(prefix, fontsize=9)
        axis.set_xlabel("round")
        axis.set_yscale("log")
    axes[0].set_ylabel("median relative error of q_t vs true median")
    axes[0].legend(fontsize=8)
    fig.suptitle("Private median tracking on logged per-client update norms (10% of the budget)", fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    path = os.path.join(HERE, "figures", "private_dispersion_tracking.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    print(check_one_round()[0])
    print(accounting_table()[0])
    path, _, curves = tracking()
    print(path)
    print(figure(curves))


if __name__ == "__main__":
    main()
