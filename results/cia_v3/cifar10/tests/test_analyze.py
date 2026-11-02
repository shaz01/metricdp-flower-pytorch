"""P2 stage-a analysis on synthetic run folders (no runs, no GPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from results.cia_v2.randomized_inclusion.runner import CIFAR10_CANDIDATES, CIFAR10_TOP_DECILE, design_matrix
from results.cia_v3.cifar10 import analyze

ROUNDS = list(range(0, 51, 5))


def write_runs(root, replicates, *, total=32, design_seed=2, gap=0.5, seed=0, flip=None):
    """Fake runner output: IN candidates have shadow loss lower by ``gap``; ``flip`` = (candidate row, replicate)."""
    rng = np.random.default_rng(seed)
    design = design_matrix(total, design_seed=design_seed)
    for j in replicates:
        included = [c for c, member in zip(CIFAR10_CANDIDATES, design[:, j]) if member]
        if flip is not None and flip[1] == j:
            candidate = CIFAR10_CANDIDATES[flip[0]]
            included = [c for c in included if c != candidate] if candidate in included else included + [candidate]
        name = f"ri-vanilla-d{design_seed}-rep{j:02d}"
        folder = root / name
        folder.mkdir(parents=True)
        manifest = {"config": "vanilla", "replicate": j, "run_name": name, "design_seed": design_seed,
                    "replicates": total, "checkpoint_rounds": ROUNDS, "candidates": list(CIFAR10_CANDIDATES),
                    "top_decile": list(CIFAR10_TOP_DECILE), "included": included}
        (folder / "manifest.json").write_text(json.dumps(manifest))
        rows = [{"round": t, "candidate": c, "test_accuracy": 0.6,
                 "shadow_loss": 2.0 - t / 50 - gap * (c in included) * t / 50 + rng.normal(0, 0.1)}
                for t in ROUNDS for c in CIFAR10_CANDIDATES]
        (folder / "measurements.json").write_text(json.dumps(rows))
        (folder / f"{name}.json").write_text(json.dumps({"client_evaluate_metrics": {
            "50": {"per-client-eval_acc": [0.4, 0.7]}}}))
        (folder / "complete.json").write_text("{}")


def test_loads_the_first_replicates_against_the_design(tmp_path):
    write_runs(tmp_path, range(10))
    audit = analyze.load_first_replicates(tmp_path, config="vanilla", replicates=8)
    assert audit.losses.shape == (32, 8, 11)
    assert np.array_equal(audit.membership, design_matrix(32, design_seed=2)[:, :8])
    assert audit.min_client_accuracy.tolist() == [0.4] * 8
    assert audit.candidates == tuple(CIFAR10_CANDIDATES)


def test_refuses_missing_incomplete_or_off_design_runs(tmp_path):
    write_runs(tmp_path / "missing", range(7))
    with pytest.raises(ValueError, match="missing"):
        analyze.load_first_replicates(tmp_path / "missing", config="vanilla", replicates=8)
    write_runs(tmp_path / "partial", range(8))
    next((tmp_path / "partial").glob("*rep03/complete.json")).unlink()
    with pytest.raises(ValueError, match="incomplete"):
        analyze.load_first_replicates(tmp_path / "partial", config="vanilla", replicates=8)
    write_runs(tmp_path / "flipped", range(8), flip=(0, 5))
    with pytest.raises(ValueError, match="differs from design"):
        analyze.load_first_replicates(tmp_path / "flipped", config="vanilla", replicates=8)


def test_g_p2a_passes_with_signal_and_fails_without(tmp_path):
    write_runs(tmp_path / "signal", range(8), gap=0.5)
    write_runs(tmp_path / "none", range(8), gap=0.0)
    for case, verdict in (("signal", "pass"), ("none", "fail")):
        audit = analyze.load_first_replicates(tmp_path / case, config="vanilla", replicates=8)
        summary = analyze.m2.config_summary(audit, noise_ratio=0.0, resamples=50)
        assert analyze.g_p2a(summary)["verdict"].startswith(verdict)


def test_g_p2a_threshold_is_inclusive():
    assert analyze.g_p2a({"all": {"auc": {"estimate": 0.65}}})["verdict"] == "pass"
    assert analyze.g_p2a({"all": {"auc": {"estimate": 0.6499}}})["verdict"].startswith("fail")


def test_safeguard():
    assert analyze.safeguard(0.55, 0.33)
    assert not analyze.safeguard(0.55, 0.29)
    assert not analyze.safeguard(0.34, 0.33)


def test_window_takes_the_smallest_qualifying_ratio():
    grid = {1: 0.62, 2: 0.60, 4: 0.50, 8: 0.30, 16: 0.10}
    assert analyze.choose_window(grid, t2=0.55, t4=0.33) == {"levels": [2, 4, 8], "qualifies": True, "shortfall": 0}
    grid[4] = 0.56  # now (4, 8, 16) also qualifies; the smaller ratio still wins
    assert analyze.choose_window(grid, t2=0.55, t4=0.33)["levels"] == [2, 4, 8]


def test_window_falls_back_to_the_smallest_shortfall_then_the_smaller_ratio():
    grid = {1: 0.61, 2: 0.60, 4: 0.54, 8: 0.3267, 16: 0.10}
    chosen = analyze.choose_window(grid, t2=0.5536, t4=0.3263)
    assert chosen["levels"] == [2, 4, 8] and not chosen["qualifies"]
    assert chosen["shortfall"] == pytest.approx(0.0004)
    tied = {1: 0.50, 2: 0.50, 4: 0.50, 8: 0.50}
    assert analyze.choose_window(tied, t2=0.55, t4=0.33)["levels"] == [1, 2, 4]
