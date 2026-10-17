"""M0 escalation support: why the v2 design cannot resolve 0.05 AUC, and what the levers buy (simulation only).

Same empirical model as ``power.py`` (WP2 vanilla seeds 42-45). Two outputs in ``tables/``:

- ``power_diagnosis.csv``: per R, the true replicate-to-replicate SD of the AUC difference between two configurations
  (baseline scale 1, weaker scale 1 or 0.8), the mean half-width of the paired two-way bootstrap interval, and the
  difference detectable at 80% power with an exact interval (2.8 SD).
- ``power_options.csv``: per number of candidates C (16 as specified, or 32) and R (16, 32), the false-positive rate
  (equal configurations) and the power at a ~0.055 difference (scale 0.8), for the paired two-way bootstrap
  (measurement standard, item 5) and a replicate-only bootstrap that conditions on the fixed candidate panel.
  With 32 candidates the top-decile share is kept (10 of 32) and shifts cycle over the WP2 sources as in ``power.py``.

    uv run python -m results.cia_v2.measurement.power_options
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.cia.audit_attacks import calibrated_scores, difference_bootstrap, roc_auc
from experiments.cia.inclusion import balanced_design
from results.cia_v2.measurement import power

SIMULATIONS, RESAMPLES = 150, 200


def sources(candidates):
    top = int(round(candidates * 5 / 16))
    return [power.TOP_SOURCES[i % 2] for i in range(top)] + [power.OTHER_SOURCES[i % 3] for i in range(candidates - top)]


def simulate(model, scale, membership, rng):
    names = sources(membership.shape[0])
    out = np.array([model["means"][("out", t)] for t in names])
    shift = np.array([model["means"][("in", t)] for t in names]) - out
    noise = rng.multivariate_normal(np.zeros(len(power.ROUNDS)), model["covariance"], size=membership.shape)
    return out[:, None, :] + scale * membership[..., None] * shift[:, None, :] + noise


def replicate_only_interval(a, member_a, b, member_b, resamples, seed):
    rng = np.random.default_rng(seed)
    replicates = a.shape[1]
    values = []
    for _ in range(resamples):
        j = rng.integers(replicates, size=replicates)
        values.append(roc_auc(a[:, j], member_a[:, j]) - roc_auc(b[:, j], member_b[:, j]))
    return np.percentile(values, [2.5, 97.5])


def run_pair(model, candidates, replicates, relative, rng, seed):
    membership = balanced_design(candidates, replicates, seed=int(rng.integers(2**31)))
    a = calibrated_scores(simulate(model, 1.0, membership, rng), membership)
    b = calibrated_scores(simulate(model, relative, membership, rng), membership)
    two_way = difference_bootstrap(a, membership, b, membership, roc_auc, resamples=RESAMPLES, seed=seed)
    low, high = replicate_only_interval(a, membership, b, membership, RESAMPLES, seed)
    return two_way, (low, high)


def main():
    model = power.empirical_model(power.load_wp2())
    rng = np.random.default_rng(11)
    diagnosis = []
    for replicates in (8, 16, 32):
        for relative in (1.0, 0.8):
            pairs = [run_pair(model, 16, replicates, relative, rng, s)[0] for s in range(SIMULATIONS)]
            sd = float(np.std([p.estimate for p in pairs]))
            half = float(np.mean([(p.high - p.low) / 2 for p in pairs]))
            diagnosis.append(dict(candidates=16, replicates=replicates, relative_scale=relative,
                                  mean_difference=float(np.mean([p.estimate for p in pairs])), true_sd=sd,
                                  two_way_half_width=half, half_width_in_sd=half / sd, exact_interval_mde=2.8 * sd))
            print(diagnosis[-1], flush=True)
    pd.DataFrame(diagnosis).to_csv(power.OUT / "power_diagnosis.csv", index=False)
    rng = np.random.default_rng(3)
    options = []
    for candidates in (16, 32):
        for replicates in (16, 32):
            row = dict(candidates=candidates, replicates=replicates)
            for relative, label in ((1.0, "null"), (0.8, "effect")):
                pairs = [run_pair(model, candidates, replicates, relative, rng, s) for s in range(SIMULATIONS)]
                row[f"{label}_mean_difference"] = float(np.mean([p[0].estimate for p in pairs]))
                row[f"{label}_rate_two_way"] = float(np.mean([p[0].low > 0 or p[0].high < 0 for p in pairs]))
                row[f"{label}_rate_replicate_only"] = float(np.mean([p[1][0] > 0 or p[1][1] < 0 for p in pairs]))
            options.append(row)
            print(row, flush=True)
    pd.DataFrame(options).to_csv(power.OUT / "power_options.csv", index=False)


if __name__ == "__main__":
    main()
