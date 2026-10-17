"""G-M1 verdict (ROADMAP.md Section 3, M1): thresholds on the 4x decoupled paired score."""

from __future__ import annotations

from pathlib import Path

import pytest

from results.cia_v2.measurement.m1.analyze import evaluate, verdict


@pytest.mark.parametrize("score,expected", [
    (0.55, "confirmed"), (0.70, "confirmed"), (0.701, "partial"), (0.749, "partial"),
    (0.75, "not material"), (0.80, "not material"),
])
def test_verdict_follows_the_fixed_table(score, expected):
    assert verdict(score) == expected


STEP3 = Path(__file__).resolve().parents[5] / "results/cia_frontier/consensus_kl/results/step3"


@pytest.mark.skipif(not STEP3.exists(), reason="v1 step-3 trajectories not checked out")
def test_reproduces_v1_step3_on_the_coupled_trajectories():
    report = evaluate(STEP3, resamples=50)["cells"]
    assert report["gdp-r0.001546"]["paired_score"] == pytest.approx(0.788)
    assert report["gdp-r0.006184"]["paired_score"] == pytest.approx(0.792)
    correlations = report["gdp-r0.006184"]["noise_energy_correlation_in_vs_out"].values()
    assert min(correlations) > 0.9  # v1 shared the noise draws (REVIEW.md Section 3)
