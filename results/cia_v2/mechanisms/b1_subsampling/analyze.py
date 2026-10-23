"""B1 analysis: client subsampling against full-participation global DP at matched accuracy (STATE.md).

Per sampling rate ``q``, the AUC-accuracy curve of its levels is compared with M2's global-DP curve at the targets
(A1's matched-accuracy statistic and intervals, ``results/cia_v2/frontier/a1/analyze.py``). The formal side compares
the amplified ``mu`` (``accounting.dominating_mu`` of the Poisson-subsampled Gaussian, noise multiplier ``r q N``) with
global DP's ``sqrt(T) / (N r)``, each interpolated in accuracy (log ``mu``) between its levels. Where the accountant
overflows (privacy losses beyond floating range at these small noise multipliers) the amplified ``mu`` is reported as
not computable, never as a number.

    uv run --no-sync python -m results.cia_v2.mechanisms.b1_subsampling.analyze \\
        --gdp gdp-r1x=0.001546 ... --rate 0.5 b1-q0.5-r...=<ratio> ... --rate 0.25 ... --out .../analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from metricdp_pytorch import accounting
from results.cia_v2.frontier import analyze as m2
from results.cia_v2.frontier.a1 import analyze as a1
from results.cia_v2.randomized_inclusion.analysis import load_audit

HERE = Path(__file__).resolve().parent
REDUCTION = 0.05  # G-B1 as amended by owner decision 9
DISCRETIZATION = 0.1  # coarse PLD grid: still a pessimistic upper bound, and feasible at these noise multipliers


def amplified_mu(ratio, sampling_rate, active=m2.ACTIVE, rounds=m2.ROUNDS, discretization=DISCRETIZATION):
    """Dominating ``mu`` (delta floor 1e-10) of the subsampled Gaussian, or ``inf`` where the accountant overflows."""
    mechanism = accounting.SubsampledGaussian(ratio * active * sampling_rate, sampling_rate, rounds)
    with np.errstate(all="ignore"):
        mu = accounting.dominating_mu(mechanism, discretization=discretization).mu
    return mu if math.isfinite(mu) else math.inf


def mu_at(points, target):
    """``mu`` at ``target`` accuracy, linear in log ``mu`` between (accuracy, mu) points; NaN outside, inf if any
    bracketing point is inf."""
    accuracy, mu = np.array(sorted(points)).T
    if not accuracy[0] <= target <= accuracy[-1]:
        return math.nan
    k = int(np.searchsorted(accuracy, target))
    lo, hi = max(k - 1, 0), min(k, len(accuracy) - 1)
    if not (math.isfinite(mu[lo]) and math.isfinite(mu[hi])):
        return math.inf
    return float(np.exp(np.interp(target, accuracy, np.log(mu))))


def g_b1(matched_all, mu_b1, mu_gdp):
    """Pass: AUC at least 0.05 lower with the gate interval below 0, and amplified ``mu`` no larger than global DP's."""
    difference = matched_all["difference"]
    auc_ok = (difference["estimate"] is not None and difference["high"] is not None
              and difference["estimate"] <= -REDUCTION and difference["high"] < 0)
    if math.isnan(mu_b1) or math.isnan(mu_gdp):
        mu_ok = None
    else:
        mu_ok = mu_b1 <= mu_gdp
    return {"verdict": "pass" if auc_ok and mu_ok else "fail", "auc_condition": bool(auc_ok),
            "mu_condition": mu_ok, "mu_b1": None if math.isinf(mu_b1) or math.isnan(mu_b1) else mu_b1,
            "mu_b1_computable": math.isfinite(mu_b1), "mu_gdp": None if math.isnan(mu_gdp) else mu_gdp}


def rate_report(audits, ratios, sampling_rate, gdp, gdp_points, targets, resamples=m2.RESAMPLES):
    mus = [amplified_mu(r, sampling_rate, rounds=a.rounds[-1]) for a, r in zip(audits, ratios)]
    levels = []
    for audit, ratio, mu in zip(audits, ratios, mus):
        summary = m2.config_summary(audit, noise_ratio=ratio, resamples=resamples)
        levels.append(a1._json(summary | {"formal_mu": None if math.isinf(mu) else mu, "formal_mu_computable":
                                          math.isfinite(mu), "sampling_rate": sampling_rate}))
    matched = a1.mechanism_report(audits, gdp, targets=targets, resamples=resamples)
    b1_points = [(float(a.final_accuracy.mean()), mu) for a, mu in zip(audits, mus)]
    gates = {str(t): g_b1(matched["all"][str(t)], mu_at(b1_points, t), mu_at(gdp_points, t)) for t in targets}
    return {"levels": levels, "matched": matched, "g_b1": gates}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdp-root", type=Path, default=HERE.parents[1] / "frontier" / "results")
    parser.add_argument("--root", type=Path, default=HERE / "results")
    parser.add_argument("--gdp", nargs="+", required=True, help="M2 configurations as config=noise_ratio")
    parser.add_argument("--rate", nargs="+", action="append", required=True,
                        help="a sampling rate followed by its configurations as config=noise_ratio")
    parser.add_argument("--targets", nargs="+", type=float, default=list(a1.TARGETS))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    gdp_levels = a1._parse(args.gdp)
    gdp = [load_audit(args.gdp_root, config=name) for name, _ in gdp_levels]
    gdp_points = [(float(a.final_accuracy.mean()), m2.formal_mu(r, rounds=a.rounds[-1]))
                  for a, (_, r) in zip(gdp, gdp_levels)]
    report = {"targets": args.targets, "rates": {}}
    for group in args.rate:
        rate, items = float(group[0]), a1._parse(group[1:])
        audits = [load_audit(args.root, config=name) for name, _ in items]
        report["rates"][str(rate)] = rate_report(audits, [r for _, r in items], rate, gdp, gdp_points,
                                                 tuple(args.targets))
    text = json.dumps(report, indent=2, allow_nan=False, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
