"""C1 analysis (synthetic; no training)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from experiments.cia.inclusion import balanced_design
from results.cia_v2.composite import analyze
from results.cia_v2.randomized_inclusion.analysis import Audit
from results.cia_v2.randomized_inclusion.runner import CANDIDATES

MEMBERSHIP = balanced_design(len(CANDIDATES), 32, seed=0)


def audit(shift, accuracy, seed, adjusted_shift=None, surrogate_shift=None):
    rng = np.random.default_rng(seed)

    def losses(s):
        return rng.normal(size=(len(CANDIDATES), 32, 11)) - s * MEMBERSHIP[..., None]

    return Audit(config="x", rounds=tuple(range(0, 51, 5)), losses=losses(shift), membership=MEMBERSHIP,
                 final_accuracy=accuracy + rng.normal(0, 0.005, 32), released_norms=None,
                 surrogate_losses=None if surrogate_shift is None else losses(surrogate_shift),
                 run_names=tuple(map(str, range(32))),
                 adjusted_losses=None if adjusted_shift is None else losses(adjusted_shift))


def curve(points, seed, scale=1.0, adjusted=None, surrogate=None):
    """Leakage shift ``scale * 2 (accuracy - 0.5)``; the adaptive views get their own scales (scaling, not shifting,
    keeps the signal's sign, which the calibrated attacker would otherwise learn either way)."""
    return [audit(scale * 2 * (a - 0.5), a, seed + i,
                  None if adjusted is None else adjusted * 2 * (a - 0.5),
                  None if surrogate is None else surrogate * 2 * (a - 0.5)) for i, a in enumerate(points)]


def references():
    gdp = curve([0.80, 0.74, 0.66, 0.54], 100, surrogate=0.9)
    return {"metric_privacy": curve([0.76, 0.68, 0.61], 200), "private_median": curve([0.75, 0.68, 0.62], 300),
            "global_dp": gdp, "subsampling_q0.25": curve([0.75, 0.70, 0.60], 400, scale=0.6)}, gdp


def test_a_composite_well_below_metric_privacy_passes():
    refs, gdp = references()
    composite = curve([0.76, 0.70, 0.62], 500, scale=0.2, adjusted=0.2, surrogate=0.1)
    result = analyze.report(composite, [0.0001, 0.003, 0.006], refs, gdp, (0.74, 0.66), resamples=20)
    assert {g["verdict"] for g in result["g_c1"].values()} == {"pass"}
    assert set(result["matched"]) == {"metric_privacy", "private_median", "global_dp", "subsampling_q0.25"}
    assert set(result["adaptive"]) == {"defense_aware_vs_metric_privacy", "surrogate_vs_global_dp_surrogate"}
    assert "adjusted_attacker" in result["levels"][0]
    json.dumps(result, allow_nan=False, default=str)


def test_a_defense_aware_attacker_that_recovers_the_leak_is_visible():
    refs, gdp = references()
    composite = curve([0.76, 0.70, 0.62], 600, scale=0.2, adjusted=1.0, surrogate=0.1)
    result = analyze.report(composite, [0.0001, 0.003, 0.006], refs, gdp, (0.74, 0.66), resamples=20)
    gated = result["matched"]["metric_privacy"]["all"]["0.74"]["difference"]["estimate"]
    aware = result["adaptive"]["defense_aware_vs_metric_privacy"]["all"]["0.74"]["difference"]["estimate"]
    assert gated < -0.05 and aware > gated + 0.05


def test_a_composite_like_metric_privacy_fails():
    refs, gdp = references()
    composite = curve([0.76, 0.70, 0.62], 700, adjusted=1.0, surrogate=1.0)
    result = analyze.report(composite, [0.0001, 0.003, 0.006], refs, gdp, (0.74, 0.66), resamples=20)
    assert {g["verdict"] for g in result["g_c1"].values()} == {"fail"}


def test_missing_adaptive_scores_are_refused():
    with pytest.raises(ValueError, match="adjusted_losses"):
        analyze.with_losses([audit(0.5, 0.7, 1)], "adjusted_losses")
