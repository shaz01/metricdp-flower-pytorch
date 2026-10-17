"""M0 power table (ROADMAP.md Section 3, M0 item 6): minimum detectable AUC difference for R = 8, 12, 16.

Empirical inputs (no training): WP2's fixed-layout vanilla runs at seeds 42-45 (EuroSAT, alpha 0.3,
48 clients, 50 rounds, no DP, seed-42 layout). For each WP2 target (23, 7, 1, 0, 3) the four IN
and four OUT-target trajectories have independent training seeds; they give per-target IN and
OUT mean shadow-loss trajectories at rounds 5, 10, ..., 50 and one replicate-to-replicate
covariance pooled over targets and worlds (Ledoit-Wolf). Round 0 is left out: v2 shares the
initial model between replicates, so its round-0 loss is constant and carries no signal.

Simulation of one v2 audit: 16 candidates in ``runner.CANDIDATES`` order; the five top-decile
candidates take the shifts of WP2's two top-decile targets (23, 7 alternating), the others those
of targets 1, 0, 3 in turn. Loss = OUT mean + inclusion x scale x (IN mean - OUT mean) + Gaussian
noise with the pooled covariance. Two configurations of one design (baseline scale ``a``, weaker
scale ``b``) are scored with the primary attacker (``calibrated_scores``) and compared with the
paired two-way bootstrap (``difference_bootstrap``); a difference counts as detected when its
95% interval excludes 0. The noise of the two configurations is drawn independently
(conservative: real configurations share replicate seeds). The minimum detectable difference is
the expected AUC difference at which the detection rate reaches 80%, interpolated over scales.

Run from the repository root:
    uv run python -m results.cia_v2.measurement.power
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import ledoit_wolf_shrinkage

from experiments.cia.audit_attacks import calibrated_scores, difference_bootstrap, roc_auc
from experiments.cia.inclusion import balanced_design
from results.cia_v2.randomized_inclusion.runner import CANDIDATES, TOP_DECILE

ROOT = Path(__file__).resolve().parents[3]
WP2 = ROOT / "results/cia_frontier/signature_suppression/results/stage_a/vanilla"
OUT = Path(__file__).resolve().parent / "tables"
WP2_TARGETS = (23, 7, 1, 0, 3)
WP2_SEEDS = (42, 43, 44, 45)
ROUNDS = tuple(range(5, 51, 5))
TOP_SOURCES, OTHER_SOURCES = (23, 7), (1, 0, 3)


def load_wp2(directory=WP2):
    """``{(world, target, seed): loss trajectory at ROUNDS}``, world "in" or "out"."""
    trajectories = {}
    for folder in sorted(p for p in directory.iterdir() if (p / "complete.json").exists()):
        seed = int(re.search(r"__seed-(\d+)__", folder.name).group(1))
        out = re.search(r"-out-(\d+)-", folder.name)
        rows = json.loads((folder / "measurements.json").read_text())
        for target in ([int(out.group(1))] if out else WP2_TARGETS):
            loss = {r["round"]: r["clean_loss"] for r in rows if r["target"] == target}
            trajectories[("out" if out else "in", target, seed)] = np.array([loss[r] for r in ROUNDS])
    return trajectories


def empirical_model(trajectories, targets=WP2_TARGETS, seeds=WP2_SEEDS):
    """Per-target IN and OUT means and the pooled, shrunk within-world covariance."""
    means, residuals = {}, []
    for target in targets:
        for world in ("in", "out"):
            group = np.array([trajectories[(world, target, seed)] for seed in seeds])
            means[(world, target)] = group.mean(axis=0)
            residuals.append(group - group.mean(axis=0))
    residuals = np.concatenate(residuals)
    sample = residuals.T @ residuals / (len(residuals) - 2 * len(targets))
    shrinkage = ledoit_wolf_shrinkage(residuals, assume_centered=True)
    k = sample.shape[0]
    covariance = (1 - shrinkage) * sample + shrinkage * np.trace(sample) / k * np.eye(k)
    return {"means": means, "covariance": covariance, "shrinkage": float(shrinkage)}


def sources():
    """WP2 target whose shift each v2 candidate takes."""
    top, other = iter(TOP_SOURCES * 3), iter(OTHER_SOURCES * 4)
    return [next(top) if c in TOP_DECILE else next(other) for c in CANDIDATES]


def simulate(model, scale, membership, rng):
    out = np.array([model["means"][("out", t)] for t in sources()])
    shift = np.array([model["means"][("in", t)] for t in sources()]) - out
    noise = rng.multivariate_normal(np.zeros(len(ROUNDS)), model["covariance"], size=membership.shape)
    return out[:, None, :] + scale * membership[..., None] * shift[:, None, :] + noise


def detection(model, replicates, scale_a, scale_b, simulations, resamples, seed):
    """Per subset: mean AUC of each configuration and the detection rate of their difference."""
    rng = np.random.default_rng(seed)
    top = np.isin(CANDIDATES, TOP_DECILE)
    records = {"all": [], "top-decile": []}
    for s in range(simulations):
        membership = balanced_design(len(CANDIDATES), replicates, seed=int(rng.integers(2**31)))
        a = calibrated_scores(simulate(model, scale_a, membership, rng), membership)
        b = calibrated_scores(simulate(model, scale_b, membership, rng), membership)
        for subset, rows in (("all", slice(None)), ("top-decile", top)):
            result = difference_bootstrap(a[rows], membership[rows], b[rows], membership[rows], roc_auc,
                                          resamples=resamples, seed=s)
            records[subset].append((roc_auc(a[rows], membership[rows]), roc_auc(b[rows], membership[rows]),
                                    result.low > 0 or result.high < 0))
    return {subset: dict(auc_a=float(np.mean([r[0] for r in rows])), auc_b=float(np.mean([r[1] for r in rows])),
                         detected=float(np.mean([r[2] for r in rows])))
            for subset, rows in records.items()}


def minimum_detectable(grid):
    """Expected AUC difference at 80% detection, linear interpolation along increasing difference."""
    grid = grid.sort_values("difference")
    above = grid[grid.detected >= 0.8]
    if above.empty:
        return float("nan")
    i = grid.index.get_loc(above.index[0])
    if i == 0:
        return float(grid.difference.iloc[0])
    lo, hi = grid.iloc[i - 1], grid.iloc[i]
    return float(lo.difference + (0.8 - lo.detected) * (hi.difference - lo.difference) / (hi.detected - lo.detected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--baselines", type=float, nargs="+", default=[1.0, 0.5])
    parser.add_argument("--relative-scales", type=float, nargs="+", default=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0])
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--resamples", type=int, default=300)
    args = parser.parse_args()
    model = empirical_model(load_wp2())
    rows = []
    for replicates in args.replicates:
        for baseline in args.baselines:
            for relative in args.relative_scales:
                result = detection(model, replicates, baseline, baseline * relative, args.simulations,
                                   args.resamples, seed=1000 * replicates + int(100 * baseline) + int(10 * relative))
                for subset, values in result.items():
                    rows.append(dict(replicates=replicates, baseline_scale=baseline, relative_scale=relative,
                                     subset=subset, **values, difference=values["auc_a"] - values["auc_b"]))
                print(rows[-2], rows[-1], flush=True)
    grid = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    grid.to_csv(OUT / "power_grid.csv", index=False)
    summary = (grid.groupby(["replicates", "baseline_scale", "subset"])
               .apply(lambda g: pd.Series({"baseline_auc": g.auc_a.mean(), "mde_auc": minimum_detectable(g)}),
                      include_groups=False)
               .reset_index())
    summary.to_csv(OUT / "power_table.csv", index=False)
    (OUT / "power_model.json").write_text(json.dumps({
        "source": str(WP2.relative_to(ROOT)), "targets": WP2_TARGETS, "seeds": WP2_SEEDS, "rounds": ROUNDS,
        "shrinkage": model["shrinkage"], "simulations": args.simulations, "resamples": args.resamples,
        "shift_norms": {str(t): float(np.linalg.norm(model["means"][("in", t)] - model["means"][("out", t)]))
                        for t in WP2_TARGETS},
        "noise_sd_per_round": np.sqrt(np.diag(model["covariance"])).round(5).tolist(),
    }, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
