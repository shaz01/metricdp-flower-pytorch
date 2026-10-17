"""Calibration of the gate interval's confidence level (ESCALATION.md resolution, item 2).

Same model and design as ``power.py`` (32 candidates, R = 32, WP2 vanilla seeds 42-45). For each baseline effect
(vanilla-sized and half) and relative scale 1 (no difference) or 0.8 (a ~0.05 difference), 400 simulated pairs of
configurations; the leave-one-pair-out jackknife standard error is computed once per pair and the interval is read
off at several confidence levels. Output ``tables/interval_calibration.csv``: false-positive rate and power per level,
for all candidates and the top decile. The gate level is the lowest level whose false-positive rate is at most 5% in
every all-candidate cell.

    uv run python -m results.cia_v2.measurement.interval_calibration
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student

from experiments.cia.audit_attacks import calibrated_scores, roc_auc
from experiments.cia.inclusion import balanced_design
from results.cia_v2.measurement import power
from results.cia_v2.randomized_inclusion.runner import CANDIDATES, TOP_DECILE

LEVELS = (0.95, 0.975, 0.99)
SIMULATIONS = 400
REPLICATES = 32


def jackknife(a, member_a, b, member_b):
    pairs = REPLICATES // 2
    keep = [np.setdiff1d(np.arange(REPLICATES), [2 * p, 2 * p + 1]) for p in range(pairs)]
    values = np.array([roc_auc(a[:, k], member_a[:, k]) - roc_auc(b[:, k], member_b[:, k]) for k in keep])
    estimate = roc_auc(a, member_a) - roc_auc(b, member_b)
    return estimate, float(np.sqrt((pairs - 1) / pairs * np.sum((values - values.mean()) ** 2)))


def main():
    model = power.empirical_model(power.load_wp2())
    top = np.isin(CANDIDATES, TOP_DECILE)
    rows = []
    for baseline in (1.0, 0.5):
        for relative in (1.0, 0.8):
            rng = np.random.default_rng(int(1000 * baseline + 10 * relative))
            draws = {"all": [], "top-decile": []}
            for _ in range(SIMULATIONS):
                membership = balanced_design(len(CANDIDATES), REPLICATES, seed=int(rng.integers(2**31)))
                a = calibrated_scores(power.simulate(model, baseline, membership, rng), membership)
                b = calibrated_scores(power.simulate(model, baseline * relative, membership, rng), membership)
                for subset, keep in (("all", slice(None)), ("top-decile", top)):
                    draws[subset].append(jackknife(a[keep], membership[keep], b[keep], membership[keep]))
            for subset, values in draws.items():
                estimates, errors = np.array(values).T
                row = dict(baseline_scale=baseline, relative_scale=relative, subset=subset,
                           mean_difference=float(estimates.mean()), sd_difference=float(estimates.std()),
                           mean_jackknife_se=float(errors.mean()))
                for level in LEVELS:
                    q = student.ppf(0.5 + level / 2, REPLICATES // 2 - 1)
                    row[f"rejection_rate_{level:g}"] = float(np.mean(np.abs(estimates) > q * errors))
                rows.append(row)
                print(row, flush=True)
    pd.DataFrame(rows).to_csv(power.OUT / "interval_calibration.csv", index=False)


if __name__ == "__main__":
    main()
