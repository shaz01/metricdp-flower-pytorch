"""A1 analysis: metric privacy against global DP at matched accuracy (STATE.md, Attacker and metrics).

Each mechanism's leakage is a curve of AUC against mean final accuracy, linear between its noise levels; global DP's
curve comes from M2's runs. The statistic at a target accuracy ``a*`` is ``AUC_mechanism(a*) - AUC_GDP(a*)``; a target
outside a curve's accuracy range is not extrapolated. Every configuration shares M2's design (``design_seed = 2``),
so each resample draws the same pairs, candidates and replicates in all of them and recomputes both curves:

- gate interval: leave-one-pair-out jackknife at ``GATE_LEVEL``;
- reference: two-way bootstrap over candidates and replicates.

Scores are the calibrated attacker's, computed once per configuration and subset as in M2.

    uv run --no-sync python -m results.cia_v2.frontier.a1.analyze \\
        --gdp gdp-r1x=0.001546 ... --metric-privacy a1-mp-...=<ratio> ... --private-median a1-pm-...=<ratio> ... \\
        --out results/cia_v2/frontier/a1/analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.cia.audit_attacks import (
    GATE_LEVEL, _complementary_pairs, _jackknife_interval, _resampled_indices, calibrated_scores, roc_auc,
)
from metricdp_pytorch.private_metric_strategy import bit_noise_for_share, privacy_account
from results.cia_v2.frontier import analyze as m2
from results.cia_v2.randomized_inclusion.analysis import load_audit
from results.cia_v2.randomized_inclusion.runner import CANDIDATES, MEDIAN_QUANTILE_MAX, TOP_DECILE

HERE = Path(__file__).resolve().parent
TARGETS = (0.740, 0.660)  # M2's mean accuracy at 2x and 4x, the two levels around 0.70 (FINDINGS.md)
EQUIVALENCE, DETECTABLE = 0.03, 0.10
SUBSETS = {"all": np.ones(len(CANDIDATES), dtype=bool), "top-decile": np.isin(CANDIDATES, TOP_DECILE)}


def formal_mu(privacy, ratio, active=m2.ACTIVE, rounds=m2.ROUNDS):
    """Global DP: ``sqrt(T) / (n r)``. Private median: its accountant at the runner's settings. Metric privacy: none."""
    if privacy == "global-dp":
        return m2.formal_mu(ratio, active, rounds)
    if privacy == "private-median":
        z = ratio * active
        return privacy_account(noise_multiplier=z, maximum=MEDIAN_QUANTILE_MAX,
                               bit_noise_std=bit_noise_for_share(z, MEDIAN_QUANTILE_MAX), rounds=rounds).mu_total
    return None


def curve_value(points, target):
    """AUC at ``target`` accuracy, linear between the (accuracy, AUC) points; NaN outside their accuracy range."""
    accuracy, auc = np.array(sorted(points)).T
    if not accuracy[0] <= target <= accuracy[-1]:
        return math.nan
    return float(np.interp(target, accuracy, auc))


class _Curve:
    """One curve's configurations, scored once for one candidate subset."""

    def __init__(self, audits, subset):
        self.scores = [calibrated_scores(a.losses[subset], a.membership[subset]) for a in audits]
        self.membership = [a.membership[subset] for a in audits]
        self.accuracy = [a.final_accuracy for a in audits]

    def value(self, target, index=None):
        """Curve at ``target`` on all trials, or on a resample ``index`` (candidate and replicate index arrays)."""
        points = []
        for scores, member, accuracy in zip(self.scores, self.membership, self.accuracy):
            if index is not None:
                scores, member, accuracy = scores[index], member[index], accuracy[index[1].ravel()]
            points.append((float(accuracy.mean()), roc_auc(scores.ravel(), member.ravel())))
        return curve_value(points, target)


def matched_difference(mechanism, gdp, *, target, subset=None, resamples=m2.RESAMPLES, seed=m2.BOOTSTRAP_SEED):
    """``AUC_mechanism(target) - AUC_GDP(target)`` with the gate jackknife and the two-way bootstrap."""
    subset = SUBSETS["all"] if subset is None else subset
    mech, ref = _Curve(mechanism, subset), _Curve(gdp, subset)
    candidates, replicates = mech.scores[0].shape
    rows = np.arange(candidates)[:, None]
    estimate = mech.value(target) - ref.value(target)
    pairs = _complementary_pairs(mech.membership[0])
    leave_out = [(rows, np.setdiff1d(np.arange(replicates), pair)[None, :]) for pair in pairs]
    values = np.array([mech.value(target, index) - ref.value(target, index) for index in leave_out])
    if np.isnan(values).any() or math.isnan(estimate):
        jackknife = {"estimate": estimate, "low": math.nan, "high": math.nan}
    else:
        interval = _jackknife_interval(estimate, values, GATE_LEVEL)
        jackknife = {"estimate": interval.estimate, "low": interval.low, "high": interval.high}
    draws = np.array([mech.value(target, index) - ref.value(target, index)
                      for index in _resampled_indices(candidates, replicates, resamples, seed)])
    finite = draws[np.isfinite(draws)]
    low, high = np.percentile(finite, [2.5, 97.5]) if len(finite) else (math.nan, math.nan)
    return {"target_accuracy": target, "gate_level": GATE_LEVEL,
            "mechanism_auc": mech.value(target), "gdp_auc": ref.value(target),
            "difference": jackknife,
            "difference_two_way": {"estimate": estimate, "low": float(low), "high": float(high),
                                   "resamples_bracketed": int(len(finite))}}


def outcome(interval):
    """G-A1 (STATE.md, Gate): better (interval below 0), equivalent (inside +-0.03), otherwise worse or inconclusive.

    An interval inside +-0.10 that contains 0 is "no detectable difference at this power", not equivalence."""
    low, high = interval["low"], interval["high"]
    if math.isnan(interval["estimate"]) or math.isnan(low):
        return "not bracketed"
    if high < 0:
        return "better"
    if low >= -EQUIVALENCE and high <= EQUIVALENCE:
        return "equivalent"
    if low > 0:
        return "worse"
    if low >= -DETECTABLE and high <= DETECTABLE:
        return "no detectable difference at this power"
    return "worse or inconclusive"


def _json(value):
    if isinstance(value, dict):
        return {key: _json(item) for key, item in value.items()}
    return None if isinstance(value, float) and math.isnan(value) else value


def mechanism_report(mechanism, gdp, *, targets=TARGETS, resamples=m2.RESAMPLES):
    report = {}
    for name, subset in SUBSETS.items():
        report[name] = {}
        for target in targets:
            result = matched_difference(mechanism, gdp, target=target, subset=subset, resamples=resamples)
            report[name][str(target)] = _json(result | {"outcome": outcome(result["difference"])})
    return report


def _parse(items):
    return [(name, float(ratio)) for name, ratio in (item.split("=") for item in items)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gdp-root", type=Path, default=HERE.parent / "results")
    parser.add_argument("--root", type=Path, default=HERE / "results")
    parser.add_argument("--gdp", nargs="+", required=True, help="M2 configurations as config=noise_ratio")
    parser.add_argument("--metric-privacy", nargs="+", default=[], help="config=noise_ratio")
    parser.add_argument("--private-median", nargs="+", default=[], help="config=noise_ratio")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    gdp_levels = _parse(args.gdp)
    gdp = [load_audit(args.gdp_root, config=name) for name, _ in gdp_levels]
    report = {"targets": list(TARGETS), "gdp": [_json(m2.config_summary(a, noise_ratio=r)) for a, (_, r)
                                                in zip(gdp, gdp_levels)], "mechanisms": {}}
    for privacy, items in (("metric-privacy", args.metric_privacy), ("private-median", args.private_median)):
        if not items:
            continue
        levels = _parse(items)
        audits = [load_audit(args.root, config=name) for name, _ in levels]
        summaries = []
        for audit, (_, ratio) in zip(audits, levels):
            summaries.append(_json(m2.config_summary(audit, noise_ratio=ratio) | {"formal_mu": formal_mu(privacy, ratio)}))
        report["mechanisms"][privacy] = {"levels": summaries, "matched": mechanism_report(audits, gdp)}
    text = json.dumps(report, indent=2, allow_nan=False, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
