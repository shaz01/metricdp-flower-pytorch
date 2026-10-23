"""B1 analysis (synthetic; no training)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.cia.inclusion import balanced_design
from results.cia_v2.frontier import analyze as m2
from results.cia_v2.mechanisms.b1_subsampling import analyze
from results.cia_v2.randomized_inclusion.analysis import Audit
from results.cia_v2.randomized_inclusion.runner import CANDIDATES

MEMBERSHIP = balanced_design(len(CANDIDATES), 32, seed=0)


def audit(shift, accuracy, seed):
    rng = np.random.default_rng(seed)
    losses = rng.normal(size=(len(CANDIDATES), 32, 11)) - shift * MEMBERSHIP[..., None]
    return Audit(config="x", rounds=tuple(range(0, 51, 5)), losses=losses, membership=MEMBERSHIP,
                 final_accuracy=accuracy + rng.normal(0, 0.005, 32), released_norms=None, surrogate_losses=None,
                 run_names=tuple(map(str, range(32))))


def curve(points, seed, extra=0.0):
    return [audit(2 * (a - 0.5) + extra, a, seed + i) for i, a in enumerate(points)]


def test_full_participation_mu_is_global_dps():
    assert analyze.amplified_mu(0.003092, 1.0) == pytest.approx(m2.formal_mu(0.003092))


def test_amplification_lowers_mu_at_a_moderate_noise_multiplier():
    assert analyze.amplified_mu(0.05, 0.5) < m2.formal_mu(0.05 * 0.5)  # z = 0.8: amplified below unamplified


def test_mu_interpolation_is_log_linear_and_refuses_to_extrapolate():
    points = [(0.80, 100.0), (0.70, 10.0)]
    assert analyze.mu_at(points, 0.75) == pytest.approx(math.sqrt(1000.0))
    assert math.isnan(analyze.mu_at(points, 0.85))
    assert math.isinf(analyze.mu_at([(0.80, math.inf), (0.70, 10.0)], 0.75))


def test_gate_needs_both_conditions():
    good = {"difference": {"estimate": -0.08, "low": -0.12, "high": -0.03}}
    assert analyze.g_b1(good, 20.0, 50.0)["verdict"] == "pass"
    assert analyze.g_b1(good, 60.0, 50.0)["verdict"] == "fail"
    small = {"difference": {"estimate": -0.04, "low": -0.07, "high": -0.01}}
    assert analyze.g_b1(small, 20.0, 50.0)["verdict"] == "fail"
    unknown = analyze.g_b1(good, math.inf, 50.0)
    assert unknown["verdict"] == "fail" and unknown["mu_b1_computable"] is False and unknown["mu_b1"] is None


def test_rate_report_on_synthetic_curves():
    gdp = curve([0.80, 0.74, 0.66, 0.54], seed=100)
    gdp_points = [(0.80, 143.0), (0.74, 71.0), (0.66, 36.0), (0.54, 18.0)]
    b1 = curve([0.78, 0.70, 0.62], seed=200, extra=-0.5)
    report = analyze.rate_report(b1, [0.04, 0.08, 0.16], 0.5, gdp, gdp_points, (0.74, 0.66), resamples=20)
    assert set(report["g_b1"]) == {"0.74", "0.66"}
    assert report["matched"]["all"]["0.74"]["difference"]["high"] < 0
    assert all(level["sampling_rate"] == 0.5 for level in report["levels"])
