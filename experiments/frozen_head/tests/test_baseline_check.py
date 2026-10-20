"""Baseline-strength check on synthetic features (no dataset)."""

from __future__ import annotations

import numpy as np

from experiments.frozen_head import baseline_check


def _blobs(per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 1, (10, 512))
    labels = np.repeat(np.arange(10), per_class)
    return np.abs(centers[labels] + rng.normal(0, 2, (labels.size, 512))), labels


def test_alternatives_cover_every_baseline_and_score_the_test_set():
    public_x, public_y = _blobs(2, 0)
    test_x, test_y = _blobs(30, 0)
    fits = baseline_check.fit_alternatives(public_x, public_y, test_x, test_y)
    assert set(fits) == {*baseline_check.LEGITIMATE, "oracle_c"}
    assert all(len(f["server_test_correct"]) == test_y.size for f in fits.values())
    accuracy = {name: np.mean(f["server_test_correct"]) for name, f in fits.items()}
    assert accuracy["oracle_c"] >= accuracy["wide_cv"]  # same family, L2 chosen on the scored set
    assert fits["nearest_mean"]["inverse_l2"] is None


def test_best_legitimate_ignores_oracle_and_prefers_protocol_on_ties():
    fits = {name: {"server_test_correct": [1, 0]} for name in baseline_check.LEGITIMATE}
    fits["oracle_c"] = {"server_test_correct": [1, 1]}
    assert baseline_check.best_legitimate(fits) == "protocol"
    fits["nearest_mean"] = {"server_test_correct": [1, 1]}
    assert baseline_check.best_legitimate(fits) == "nearest_mean"


def test_with_baseline_replaces_only_the_public_only_correctness():
    records = [{"job": {"public_size": 20, "public_subset": 1}, "server_test_correct": [1, 1],
                "public_only": {"server_test_correct": [0, 0], "inverse_l2": 0.01}}]
    swapped = baseline_check.with_baseline(records, {(20, 1): [1, 0]})
    assert swapped[0]["public_only"] == {"server_test_correct": [1, 0]}
    assert swapped[0]["server_test_correct"] == [1, 1] and records[0]["public_only"]["server_test_correct"] == [0, 0]
