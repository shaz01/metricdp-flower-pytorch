"""A1 analysis (synthetic; no training)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from experiments.cia.inclusion import balanced_design
from metricdp_pytorch.private_metric_strategy import bit_noise_for_share, privacy_account
from results.cia_v2.frontier.a1 import analyze
from results.cia_v2.randomized_inclusion.analysis import Audit
from results.cia_v2.randomized_inclusion.runner import CANDIDATES

REPLICATES = 32
MEMBERSHIP = balanced_design(len(CANDIDATES), REPLICATES, seed=0)  # one design shared by every configuration


def audit(shift, accuracy, *, seed):
    """Loss shift ``shift`` for members (larger: more leakage) and mean final accuracy ``accuracy``."""
    rng = np.random.default_rng(seed)
    losses = rng.normal(size=(len(CANDIDATES), REPLICATES, 11)) - shift * MEMBERSHIP[..., None]
    return Audit(config=f"s{shift}-a{accuracy}", rounds=tuple(range(0, 51, 5)), losses=losses, membership=MEMBERSHIP,
                 final_accuracy=accuracy + rng.normal(0, 0.005, REPLICATES), released_norms=None,
                 surrogate_losses=None, run_names=tuple(map(str, range(REPLICATES))))


def curve(points, *, seed, extra_shift=0.0):
    """Configurations whose leakage depends on accuracy only (shift = 2 (accuracy - 0.5)), plus ``extra_shift``."""
    return [audit(2 * (a - 0.5) + extra_shift, a, seed=seed + i) for i, a in enumerate(points)]


GDP = curve([0.80, 0.74, 0.66, 0.54], seed=100)


def test_curve_value_interpolates_in_accuracy_and_refuses_to_extrapolate():
    points = [(0.66, 0.60), (0.80, 0.74), (0.74, 0.70)]  # any order
    assert analyze.curve_value(points, 0.77) == pytest.approx(0.72)
    assert analyze.curve_value(points, 0.66) == pytest.approx(0.60)
    assert math.isnan(analyze.curve_value(points, 0.65))
    assert math.isnan(analyze.curve_value(points, 0.81))


def test_same_leakage_accuracy_relation_gives_no_difference():
    result = analyze.matched_difference(curve([0.78, 0.70, 0.62], seed=200), GDP, target=0.70, resamples=200)
    jackknife = result["difference"]
    assert jackknife["low"] <= 0 <= jackknife["high"]
    assert abs(jackknife["estimate"]) < 0.04
    assert result["difference_two_way"]["low"] <= jackknife["estimate"] <= result["difference_two_way"]["high"]
    assert analyze.outcome(jackknife) in {"equivalent", "no detectable difference at this power",
                                          "worse or inconclusive"}


def test_more_leakage_at_the_same_accuracy_is_worse():
    result = analyze.matched_difference(curve([0.78, 0.70, 0.62], seed=300, extra_shift=0.6), GDP, target=0.70,
                                        resamples=50)
    assert result["difference"]["low"] > 0
    assert analyze.outcome(result["difference"]) == "worse"


def test_less_leakage_at_the_same_accuracy_is_better():
    result = analyze.matched_difference(curve([0.78, 0.70, 0.62], seed=400, extra_shift=-0.5), GDP, target=0.70,
                                        resamples=50)
    assert result["difference"]["high"] < 0
    assert analyze.outcome(result["difference"]) == "better"


def test_a_target_outside_the_mechanism_levels_is_not_bracketed():
    result = analyze.matched_difference(curve([0.78, 0.75, 0.72], seed=500), GDP, target=0.66, resamples=20)
    assert math.isnan(result["difference"]["estimate"])
    assert analyze.outcome(result["difference"]) == "not bracketed"


@pytest.mark.parametrize("interval,expected", [
    ({"estimate": -0.08, "low": -0.12, "high": -0.01}, "better"),
    ({"estimate": 0.0, "low": -0.02, "high": 0.025}, "equivalent"),
    ({"estimate": 0.02, "low": -0.05, "high": 0.09}, "no detectable difference at this power"),
    ({"estimate": 0.02, "low": -0.12, "high": 0.16}, "worse or inconclusive"),
    ({"estimate": 0.08, "low": 0.01, "high": 0.15}, "worse"),
])
def test_outcome_follows_the_gate(interval, expected):
    assert analyze.outcome(interval) == expected


def test_private_median_formal_mu_uses_its_accountant():
    z = 0.001546 * 32
    expected = privacy_account(noise_multiplier=z, maximum=analyze.MEDIAN_QUANTILE_MAX,
                               bit_noise_std=bit_noise_for_share(z, analyze.MEDIAN_QUANTILE_MAX), rounds=50).mu_total
    assert analyze.formal_mu("private-median", 0.001546) == pytest.approx(expected)
    assert analyze.formal_mu("metric-privacy", 0.001546) is None
    assert analyze.formal_mu("global-dp", 0.001546) == pytest.approx(math.sqrt(50) / (32 * 0.001546))


def test_report_is_valid_json_and_has_both_subsets():
    mechanism = curve([0.78, 0.70, 0.62], seed=600)
    report = analyze.mechanism_report(mechanism, GDP, targets=(0.74, 0.66), resamples=20)
    assert set(report) == {"all", "top-decile"}
    assert set(report["all"]) == {"0.74", "0.66"}
    json.dumps(report, allow_nan=False, default=str)
