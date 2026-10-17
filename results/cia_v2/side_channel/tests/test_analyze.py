"""A2 analysis on fake trajectory folders (frontier runner layout plus a2_seeds.json)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from results.cia_v2.side_channel import analyze, runner


def write_cell(directory, combos, *, shift=0.3, sigma_in=0.010, sigma_out=0.012, seed=0, privacy="metric-privacy"):
    rng = np.random.default_rng(seed)
    D = 1000
    for combo in combos:
        folder = directory / combo.run_name()
        folder.mkdir(parents=True)
        inside = combo.out_target is None
        target = runner.CELLS[("alzheimer", 3)]
        (folder / "manifest.json").write_text(json.dumps({"run_name": combo.run_name(), "out_target": combo.out_target,
                                                          "targets": [target], "rounds": 50, "privacy": privacy}))
        (folder / "a2_seeds.json").write_text(json.dumps({"replicate": combo.replicate, "stage": combo.stage}))
        rows = [{"round": r, "target": target, "clean_loss": 2.0 - 0.01 * r - (shift if inside else 0) + rng.normal(0, 0.05),
                 "clean_loss_before": 2.0} for r in range(1, 51)]
        (folder / "measurements.json").write_text(json.dumps(rows))
        sigma = sigma_in if inside else sigma_out
        metrics = {str(t): {"dp-released-update-norm": float(np.sqrt(D) * sigma * (1 + rng.normal(0, 0.01))),
                            "dp-parameter-count": D, f"{'metric-dp' if privacy == 'metric-privacy' else 'global-dp'}-noise-stdv": sigma}
                   for t in range(1, 51)}
        (folder / f"{combo.run_name()}.json").write_text(json.dumps({
            "train_metrics": metrics, "server_evaluate_metrics": {"50": {"accuracy": 0.7}}}))
        (folder / "complete.json").write_text("{}")


@pytest.fixture
def cell(tmp_path):
    combos = runner.build_cell("alzheimer", 3, "metric-privacy", multiple=4, stage="main")
    write_cell(tmp_path, combos)
    return tmp_path, combos


def test_load_cell_pairs_in_and_out_by_replicate(cell):
    root, combos = cell
    data = analyze.load_cell(root)
    assert data["membership"].shape == (1, 16)
    assert data["membership"][0].tolist() == [True, False] * 8
    assert data["losses"].shape == (1, 16, 11)
    assert data["losses"][0, 0, 0] == pytest.approx(2.0)  # round 0 from clean_loss_before
    assert data["sigma_hat"].shape == (1, 16, 1)
    assert data["sigma_hat"][0, 0, 0] == pytest.approx(0.010, rel=0.01)


def test_noise_level_attacker_separates_a_noise_level_shift(cell):
    root, _ = cell
    result = analyze.cell_summary(analyze.load_cell(root))
    assert result["noise_level"]["auc"]["estimate"] > 0.95
    assert result["oracle_noise_level_auc"] == pytest.approx(1.0)
    assert result["loss"]["auc"]["estimate"] > 0.9


def test_gate_needs_two_cells():
    def cell_result(noise_low, loss_auc):
        return {"metric-privacy": {"noise_level": {"auc": {"low": noise_low}}},
                "global-dp": {"loss": {"auc": {"estimate": loss_auc}}}}
    assert analyze.g_a2({"a": cell_result(0.85, 0.55), "b": cell_result(0.81, 0.60), "c": cell_result(0.5, 0.5)})[
        "verdict"] == "strong claim"
    assert analyze.g_a2({"a": cell_result(0.85, 0.55), "b": cell_result(0.79, 0.55)})["verdict"] == "channel size recorded"
    assert analyze.g_a2({"a": cell_result(0.85, 0.61), "b": cell_result(0.9, 0.62)})["verdict"] == "channel size recorded"
