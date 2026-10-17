"""B4 calculation stage (no DP training): noise-to-signal table, predicted accuracy loss, crowd bound.

    uv run --with dp-accounting==0.6.0 python research/cia_roadmap_v2/theory/formal_regime_calcs.py [--part nsr|predict|crowd|all]

Writes ``tables/formal_regime_{nsr,predicted_accuracy,crowd}.csv`` beside this file. The
``predict`` and ``crowd`` parts need the cached frozen features (``experiments.frozen_head.features``)
and run noise-free head federations only; their settings are fixed in
``results/cia_v2/formal_regime/STATE.md``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import norm

from metricdp_pytorch import accounting as acc

TABLES = Path(__file__).resolve().parent / "tables"
HEAD_DIM = 512 * 10 + 10
FULL_NETWORK_DIM = 289_194  # EuroSAT CNN of the 48-client frontier runs (REVIEW.md Section 6)
CLIENT_COUNTS = (48, 480)
ROUNDS = (10, 20)
MU_TOTALS = (1.0, 2.0, 4.0, 8.0)


def _write(name: str, rows: list[dict]) -> None:
    TABLES.mkdir(exist_ok=True)
    with (TABLES / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {TABLES / name} ({len(rows)} rows)")


def nsr_row(regime: str, dim: int, clients: int, rounds: int, mu_total: float) -> dict:
    """Full participation, equal weights, public slot count: ``z = sqrt(T) / mu``."""
    z = acc.noise_multiplier_for_mu(mu_total, rounds=rounds)
    per_coordinate = z / clients  # noise std on the average, in units of C
    return {
        "regime": regime, "dim": dim, "clients": clients, "rounds": rounds, "mu_total": mu_total,
        "noise_multiplier": round(z, 4), "noise_std_over_C": round(per_coordinate, 6),
        "noise_norm_over_C": round(per_coordinate * math.sqrt(dim), 4),  # = NSR against a signal of norm C
        "accumulated_noise_norm_over_C": round(per_coordinate * math.sqrt(dim * rounds), 4),
        "epsilon_at_1e-5": round(acc.gdp_epsilon(mu_total, 1e-5), 3),
        # Bounds on any membership attacker against mu-GDP: AUC of G_mu and TPR = 1 - G_mu(alpha).
        "attack_auc_bound": round(float(norm.cdf(mu_total / math.sqrt(2))), 6),
        "tpr_bound_at_5pct_fpr": round(float(norm.cdf(mu_total - norm.ppf(0.95))), 6),
        "tpr_bound_at_1pct_fpr": round(float(norm.cdf(mu_total - norm.ppf(0.99))), 6),
    }


def nsr() -> list[dict]:
    rows = [nsr_row("frozen head", HEAD_DIM, n, t, mu) for n in CLIENT_COUNTS for t in ROUNDS for mu in MU_TOTALS]
    rows += [nsr_row("full network", FULL_NETWORK_DIM, 48, 50, mu) for mu in MU_TOTALS]
    return rows


# --- Parts that need the cached features -----------------------------------------------------------

TUNING_LAYOUT, EVALUATION_LAYOUT = 43, 42
SPLIT_FACTOR = {48: 1, 480: 10}
CALC_CONFIG = dict(local_epochs=5, local_lr=0.2, batch_size=32, server_lr=1.0)
MONTE_CARLO_DRAWS = 100
CROWD_OTHER_TARGETS_480 = 48  # n = 480: top decile plus this many clients spread by size


def _federation(clients: int, layout_seed: int):
    from experiments.frozen_head.data import build_layout
    from experiments.frozen_head.features import fingerprint, load_split_features
    from experiments.frozen_head.fedavg import Head
    from experiments.frozen_head.public_only import Standardizer, fit_public_only

    train_x, train_y = load_split_features("train")
    test_x, test_y = load_split_features("test")
    layout = build_layout(tuple(train_y.tolist()), tuple(test_y.tolist()), num_clients=clients,
                          layout_seed=layout_seed, split_factor=SPLIT_FACTOR[clients])
    scale = Standardizer.fit(test_x[layout.public])
    head = Head(10, 512)
    public = fit_public_only(head, scale(test_x[layout.public]), test_y[layout.public])
    train = [(scale(train_x[i]), train_y[i]) for i in layout.client_train]
    local_test = [(scale(train_x[i]), train_y[i]) for i in layout.client_test]
    return {"layout": layout, "head": head, "public": public, "train": train, "local_test": local_test,
            "fingerprint": fingerprint(train_x)}


def _calc_clip(fed: dict, clients: int) -> float:
    """Median round-0 update norm on the tuning layout under the calculation config (public init)."""
    from experiments.frozen_head.fedavg import HeadConfig, run

    probe = HeadConfig(rounds=1, clip=1e12, noise_multiplier=0.0, num_slots=clients, **CALC_CONFIG)
    return run(fed["head"], fed["public"].theta, fed["train"], probe, client_seed=0, noise_seed=0)["median_update_norm"][0]


def predict() -> list[dict]:
    """Noise-free trajectory, then the final head perturbed by the accumulated server noise."""
    from experiments.frozen_head.fedavg import HeadConfig, run

    rows = []
    for clients in CLIENT_COUNTS:
        fed = _federation(clients, TUNING_LAYOUT)
        clip = _calc_clip(fed, clients)
        pooled_x = np.concatenate([x for x, _ in fed["local_test"]])
        pooled_y = np.concatenate([y for _, y in fed["local_test"]])
        public_only = fed["head"].accuracy(fed["public"].theta, pooled_x, pooled_y)
        for rounds in ROUNDS:
            config = HeadConfig(rounds=rounds, clip=clip, noise_multiplier=0.0, num_slots=clients, **CALC_CONFIG)
            final = run(fed["head"], fed["public"].theta, fed["train"], config, client_seed=0, noise_seed=0)["theta"]
            clean = fed["head"].accuracy(final, pooled_x, pooled_y)
            rng = np.random.default_rng([clients, rounds])
            for mu in MU_TOTALS:
                z = acc.noise_multiplier_for_mu(mu, rounds=rounds)
                std = CALC_CONFIG["server_lr"] * z * clip / clients * math.sqrt(rounds)
                noisy = [fed["head"].accuracy(final + rng.normal(0, std, final.size), pooled_x, pooled_y)
                         for _ in range(MONTE_CARLO_DRAWS)]
                rows.append({
                    "clients": clients, "rounds": rounds, "mu_total": mu, "layout_seed": TUNING_LAYOUT,
                    "clip": round(clip, 4), "accumulated_noise_std": round(std, 5),
                    "public_only_acc": round(public_only, 4), "noise_free_acc": round(clean, 4),
                    "predicted_acc": round(float(np.mean(noisy)), 4), "predicted_acc_sd": round(float(np.std(noisy)), 4),
                    "noise_free_gain_pp": round(100 * (clean - public_only), 2),
                    "predicted_gain_pp": round(100 * (float(np.mean(noisy)) - public_only), 2),
                    "metric": "pooled client local-test accuracy (tuning layout)",
                })
                print(rows[-1])
    return rows


def _crowd_targets(layout, clients: int) -> tuple[np.ndarray, set[int]]:
    top = set(layout.top_decile().tolist())
    if clients == 48:
        return np.arange(clients), top
    order = np.argsort(layout.client_sizes, kind="stable")
    rest = [int(t) for t in order if int(t) not in top]
    spread = [rest[int(round((k + 0.5) * len(rest) / CROWD_OTHER_TARGETS_480))] for k in range(CROWD_OTHER_TARGETS_480)]
    return np.array(sorted(top) + spread), top


def crowd() -> list[dict]:
    """Propositions 1 and 2 per round on the noise-free evaluation-layout trajectory (update geometry only)."""
    from experiments.frozen_head.crowd import compose
    from experiments.frozen_head.fedavg import HeadConfig, run

    rows = []
    for clients in CLIENT_COUNTS:
        clip = _calc_clip(_federation(clients, TUNING_LAYOUT), clients)
        fed = _federation(clients, EVALUATION_LAYOUT)
        targets, top = _crowd_targets(fed["layout"], clients)
        for rounds in ROUNDS:
            config = HeadConfig(rounds=rounds, clip=clip, noise_multiplier=0.0, num_slots=clients, **CALC_CONFIG)
            out = run(fed["head"], fed["public"].theta, fed["train"], config, client_seed=0, noise_seed=0,
                      keep_updates=True)
            sigmas = {mu: acc.noise_multiplier_for_mu(mu, rounds=rounds) * clip / clients for mu in MU_TOTALS}
            full = {mu: np.zeros((len(targets), rounds)) for mu in MU_TOTALS}
            stat = {mu: np.zeros((len(targets), rounds)) for mu in MU_TOTALS}
            for r in range(rounds):
                updates = out["updates"][r]
                for k, target in enumerate(targets):
                    x, y = fed["local_test"][target]
                    _, direction = fed["head"].loss_and_grad(out["round_start_theta"][r], x, y)
                    _ratios_for_sigmas(updates, int(target), direction, sigmas, clip, clients, full, stat, k, r)
            for mu in MU_TOTALS:
                for group, mask in (("all", np.ones(len(targets), bool)),
                                    ("top decile", np.isin(targets, sorted(top)))):
                    per_target_full = np.array([compose(v) for v in full[mu][mask]])
                    per_target_stat = np.array([compose(v) for v in stat[mu][mask]])
                    rows.append({
                        "clients": clients, "rounds": rounds, "mu_total": mu, "targets": group,
                        "n_targets": int(mask.sum()), "layout_seed": EVALUATION_LAYOUT, "clip": round(clip, 4),
                        "full_release_ratio_median": round(float(np.median(per_target_full)), 4),
                        "full_release_ratio_max": round(float(np.max(per_target_full)), 4),
                        "loss_statistic_rho_median": round(float(np.median(per_target_stat)), 4),
                        "loss_statistic_rho_min": round(float(np.min(per_target_stat)), 4),
                    })
                    print(rows[-1])
    return rows


def _ratios_for_sigmas(updates, target, direction, sigmas, clip, clients, full, stat, k, r) -> None:
    """Both ratios for every sigma from one eigendecomposition of the leave-one-out crowd factor."""
    others = np.delete(updates, target, axis=0)
    centred = others - others.mean(axis=0)
    factor = (centred * (np.sqrt(len(others) / (len(others) - 1)) / clients)).T  # Sigma_B / n^2 = A A^T
    gram_values, gram_vectors = np.linalg.eigh(factor.T @ factor)
    u = updates[target]
    projected = gram_vectors.T @ (factor.T @ u)
    crowd_w = float(np.sum((factor.T @ direction) ** 2))
    for mu, sigma in sigmas.items():
        quadratic = (u @ u - np.sum(projected ** 2 / (sigma ** 2 + gram_values))) / sigma ** 2
        full[mu][k, r] = sigma * math.sqrt(max(quadratic, 0.0)) / clip
        noise = sigma ** 2 * float(direction @ direction)
        stat[mu][k, r] = math.sqrt(noise / (noise + crowd_w))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", choices=("nsr", "predict", "crowd", "all"), default="all")
    args = parser.parse_args()
    if args.part in ("nsr", "all"):
        _write("formal_regime_nsr.csv", nsr())
    if args.part in ("predict", "all"):
        _write("formal_regime_predicted_accuracy.csv", predict())
    if args.part in ("crowd", "all"):
        _write("formal_regime_crowd.csv", crowd())
