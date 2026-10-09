"""Frontier statistics on synthetic worlds with known separation."""

import json

import numpy as np
import pytest

from experiments.stacked_head import frontier


def make_world(directory, task, label, public_set, world, records, mix, gain=0.02):
    stem = "frontiernf" if label == "nf" else "frontier"
    risk = "q65" if label == "nf" else label
    name = f"{task}_b32_s{public_set}_A_{risk}_{stem}{world}"
    (directory / f"{name}.json").write_text(json.dumps({"summary": {"gain_over_control": gain, "accuracy_delta": 0.001}, "config": {}}))
    np.savez(directory / f"{name}.releases.npz", attack_records_gain=records, attack_gain=mix)


def test_auc_orders_distributions_and_handles_single_deterministic_releases():
    rng = np.random.default_rng(0)
    assert frontier.auc(rng.normal(1, 1, 4000), rng.normal(0, 1, 4000)) == pytest.approx(0.76, abs=0.02)   # d' = 1
    assert frontier.auc(np.array([2.0]), np.array([1.0])) == 1.0 and frontier.auc(np.array([1.0]), np.array([1.0])) == 0.5 and frontier.auc(np.array([0.0]), np.array([1.0])) == 0.0


def test_calibration_removes_a_world_wide_quality_shift_that_fools_the_raw_statistic(tmp_path):
    rng = np.random.default_rng(1)
    task, rounds = "kmnist_classes0to3", 400
    for s in range(5):
        for label in ("q65",):
            # IN releases are better everywhere (shift on both records and shadow); membership adds nothing -> calibrated AUC ~ 0.5, raw AUC high
            shift = 0.3
            in_records = rng.normal(shift, 0.1, size=(rounds, 4)); in_mix = in_records + rng.normal(0, 0.02, size=(rounds, 4))
            out_records = rng.normal(0, 0.1, size=(rounds, 4)); out_mix = out_records + rng.normal(0, 0.02, size=(rounds, 4))
            make_world(tmp_path, task, label, s, "in", in_records, in_mix)
            for t in range(4):
                make_world(tmp_path, task, label, s, f"out{t}", out_records, out_mix)
    rep = frontier.report(frontier.collect(tmp_path, (task,), tuple(range(5))))
    stats = rep["groups"][0]["statistics"]
    assert stats["raw_records"]["mean_auc"] > 0.95 and abs(stats["calibrated_records"]["mean_auc"] - 0.5) < 0.1
    assert stats["calibrated_records"]["below_target"] and rep["claim_calibration_conservative_for_practical_attack"]
    assert not stats["calibrated_records"]["detectable"]


def test_noise_free_group_is_reported_without_a_target_and_flags_detectability(tmp_path):
    task = "kmnist_classes0to3"
    for s in range(5):
        make_world(tmp_path, task, "nf", s, "in", np.full((1, 4), 0.05), np.zeros((1, 4)))
        for t in range(4):
            make_world(tmp_path, task, "nf", s, f"out{t}", np.zeros((1, 4)), np.zeros((1, 4)))
    rep = frontier.report(frontier.collect(tmp_path, (task,), tuple(range(5))))
    group = rep["groups"][0]
    assert group["target_risk"] is None and group["statistics"]["calibrated_records"]["mean_auc"] == 1.0 and group["statistics"]["calibrated_records"]["detectable"]
    assert "n/a" in frontier.format_report(rep)
