"""Runner configuration, attack statistic calibration and gate logic."""

import argparse

import numpy as np
import pytest
from scipy.special import ndtr

from experiments.stacked_head import analysis, attack, runner


def namespace(**overrides):
    parser = runner._parser()
    args = parser.parse_args(["run"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_probe_seed_matches_the_research_formula():
    assert runner.probe_seed(32, 0, "A", 0.65) == 20261012 + 32000 + 0 + 0 + 65
    assert runner.probe_seed(128, 2, "B", 0.55) == 20261012 + 128000 + 200 + 10 + 55
    assert runner.probe_seed(32, 1, "A", 0.8) == 20261012 + 32000 + 100 + 0 + 80


def test_build_run_config_single_release_and_replicate_modes():
    single = runner.build_run_config(namespace(single_release=True))
    assert single["rounds"] == 1 and single["replicate"] is False and single["run-name"].endswith("_single")
    many = runner.build_run_config(namespace(rounds=64))
    assert many["rounds"] == 64 and many["replicate"] is True and many["probe-noise-rounds"] == 64
    with pytest.raises(ValueError):
        runner.build_run_config(namespace(risk=0.5))


def test_attack_auc_matches_the_calibrated_expectation():
    rng = np.random.default_rng(0)
    d, sigma, rounds, target = 6, 0.01, 6000, 3
    contributions = rng.normal(size=(8, d)) * 0.004
    shift = contributions[target]
    peer = attack.peer_of(target)

    def world(include):
        parts = contributions.copy()
        if not include:
            parts[target] = 0
        noise = rng.normal(size=(rounds, 8, d)) * sigma / np.sqrt(7)
        return parts[None] + noise

    in_scores, out_scores = attack.attack_scores(world(True), world(False), contributions, target, sigma)
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(np.r_[np.ones(rounds), np.zeros(rounds)], np.r_[in_scores, out_scores])
    expected = float(ndtr(np.linalg.norm(shift) / (np.sqrt(2) * sigma)))
    assert peer == 7 and abs(auc - expected) < 0.02


def test_peer_of_target_seven_is_six():
    assert attack.peer_of(7) == 6 and attack.peer_of(0) == 7


def test_gate_requires_cells_threshold_mean_and_worst_cell():
    good = [0.012, 0.011, 0.01, 0.009, 0.02, 0.002]
    assert analysis.gate(good)["passed"]
    assert not analysis.gate([0.012, 0.011, 0.01, 0.009, 0.0005, 0.0005])["passed"]
    assert not analysis.gate([0.02, 0.02, 0.02, 0.02, 0.02, -0.004])["passed"]
    assert not analysis.gate(good[:5])["passed"]
