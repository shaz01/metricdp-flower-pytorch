"""B5 analysis (synthetic; no training)."""

from __future__ import annotations

import numpy as np

from experiments.cia.inclusion import balanced_design
from results.cia_v2.mechanisms.b5_large_clients import analyze
from results.cia_v2.randomized_inclusion.analysis import Audit
from results.cia_v2.randomized_inclusion.runner import CANDIDATES

MEMBERSHIP = balanced_design(len(CANDIDATES), 32, seed=0)


def audit(shift, accuracy, seed):
    rng = np.random.default_rng(seed)
    losses = rng.normal(size=(len(CANDIDATES), 32, 11)) - shift * MEMBERSHIP[..., None]
    return Audit(config="x", rounds=tuple(range(0, 51, 5)), losses=losses, membership=MEMBERSHIP,
                 final_accuracy=accuracy + rng.normal(0, 0.003, 32), released_norms=None, surrogate_losses=None,
                 run_names=tuple(map(str, range(32))))


def test_a_large_reduction_at_equal_accuracy_passes():
    difference = analyze.paired_difference(audit(0.2, 0.74, 1), audit(1.0, 0.74, 2), resamples=100)
    assert difference["all"]["difference"]["high"] < -0.05
    assert difference["all"]["difference_two_way"]["low"] <= difference["all"]["difference"]["estimate"]
    assert analyze.g_b5(difference)["verdict"] == "pass"


def test_an_accuracy_loss_above_one_point_fails():
    difference = analyze.paired_difference(audit(0.2, 0.72, 1), audit(1.0, 0.74, 2), resamples=50)
    assert difference["accuracy_difference"]["estimate"] < -0.01
    assert analyze.g_b5(difference)["verdict"] == "fail"


def test_no_reduction_fails():
    difference = analyze.paired_difference(audit(1.0, 0.74, 3), audit(1.0, 0.74, 4), resamples=50)
    assert difference["all"]["difference"]["low"] < 0 < difference["all"]["difference"]["high"]
    assert analyze.g_b5(difference)["verdict"] == "fail"
