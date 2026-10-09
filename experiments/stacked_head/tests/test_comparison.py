"""Paired comparison, verdict rule and curve interpolation on synthetic results."""

import json

import numpy as np
import pytest

from experiments.stacked_head import comparison


def write(directory, task, risk, public_set, tag, gain, fixed=None, oracle=None):
    name = f"{task}_b32_s{public_set}_A_q{int(round(risk * 100))}_{tag}"
    result = {"run_name": name, "config": {"task": task, "risk": risk, "public-set": public_set, "budget": 32, "cohort": "A"},
              "control": {"ce": 0.6, "accuracy": 0.8}, "summary": {"gain_over_control": gain, "accuracy_delta": 0.001, "pick_fraction": {}},
              "summary_fixed_multiplier_1": {"gain_over_control": gain if fixed is None else fixed, "accuracy_delta": 0.0},
              "oracle": {"per_client_mean_oracle_auc": [oracle if oracle is not None else risk] * 8, "fraction_clipped": 1.0}}
    (directory / f"{name}.json").write_text(json.dumps(result))


def populate(directory, ours_edge, tasks=("kmnist_classes0to3", "kmnist_classes4to7", "mnist_classes0to3", "mnist_classes4to7", "fmnist_classes4to7")):
    rng = np.random.default_rng(0)
    for task in tasks:
        for risk in (0.65, 0.8):
            for s in range(10):
                noise = rng.normal(0, 0.001)
                write(directory, task, risk, s, "confirmtransfer", 0.02 + ours_edge + noise)
                for tag in ("gdpconfirmtransfer", "mdpconfirmtransfer"):
                    write(directory, task, risk, s, tag, 0.02 + noise * 0.5, fixed=-0.1)
        for s in range(10):
            write(directory, task, 0.65, s, "vanconfirmtransfer", 0.1)


def test_part1_verdicts_follow_the_pre_stated_rule(tmp_path):
    populate(tmp_path, ours_edge=0.01)                      # ours is better by .01 in every group
    report = comparison.part1(tmp_path)
    for mech in ("global-dp", "metric-privacy"):
        block = report["baselines"][mech]
        assert block["verdict"] == "ours_better" and block["counts"]["ours_better"] == 10 and block["n_groups"] == 10
        g = block["groups"][0]
        assert g["mean_difference"] == pytest.approx(0.01, abs=2e-3) and g["ci95"][0] > 0 and g["baseline_ungated_mean_gain"] == pytest.approx(-0.1)
        assert g["vanilla_mean_gain"] == pytest.approx(0.1) and g["baseline_realized_oracle_auc"] == pytest.approx(g["risk"])


def test_part1_reports_baseline_better_and_mixed(tmp_path):
    populate(tmp_path, ours_edge=-0.01)
    assert comparison.part1(tmp_path)["baselines"]["global-dp"]["verdict"] == "baseline_better"
    mixed = tmp_path / "m"
    mixed.mkdir()
    populate(mixed, ours_edge=0.0)                           # equal means -> no clear difference
    assert comparison.part1(mixed)["baselines"]["global-dp"]["verdict"] == "mixed"


def test_paired_bootstrap_and_classification():
    d = np.full(10, 0.01)
    mean, low, high = comparison.paired_bootstrap(d)
    assert mean == pytest.approx(0.01) and low == pytest.approx(0.01) and comparison.classify(mean, low, high) == "ours_better"
    assert comparison.classify(0.0, -0.01, 0.01) == "no_clear_difference" and comparison.classify(-0.02, -0.03, -0.01) == "baseline_better"


def test_interpolated_gain_is_piecewise_linear_and_flags_extrapolation():
    points = [(0.5, 0.0), (0.7, 0.02), (0.9, 0.04)]
    value, outside = comparison.interpolated_gain(points, 0.6)
    assert value == pytest.approx(0.01) and not outside
    assert comparison.interpolated_gain(points, 0.95) == (pytest.approx(0.04), True)
