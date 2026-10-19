"""A2 analysis on fake trajectory folders (frontier runner layout plus a2_seeds.json)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from results.cia_v2.side_channel import analyze, runner


def write_cell(directory, combos, *, shift=0.3, sigma_in=0.010, sigma_out=0.012, seed=0, privacy="metric-privacy",
               collapsed=()):
    """``collapsed``: rounds logged as collapsed (no aggregation, no release) in every OUT trajectory."""
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
        prefix = "metric-dp" if privacy == "metric-privacy" else "global-dp"
        metrics = {str(t): {"dp-released-update-norm": float(np.sqrt(D) * sigma * (1 + rng.normal(0, 0.01))),
                            "dp-parameter-count": D, f"{prefix}-noise-stdv": sigma,
                            f"{prefix}-aggregation-collapsed": 0.0}
                   for t in range(1, 51)}
        for t in collapsed if not inside else ():
            metrics[str(t)] = {f"{prefix}-aggregation-collapsed": 1.0}
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


@pytest.mark.parametrize("auc,multiple,expected", [
    (0.60, 1, ("qualifies", 1)), (0.61, 1, ("next", 4)), (0.7, 16, ("next", 64)), (0.65, 64, ("no qualifying noise", None)),
])
def test_pilot_decision(auc, multiple, expected):
    assert analyze.pilot_decision(auc, multiple) == expected


def test_collapsed_rounds_release_nothing(tmp_path):
    """Amendment 2026-10-19: a collapsed round releases no update, so its released norm and true noise std are 0."""
    combos = runner.build_cell("alzheimer", 3, "metric-privacy", multiple=4, stage="main")
    write_cell(tmp_path, combos, sigma_in=0.010, sigma_out=0.010, collapsed=range(26, 51))
    data = analyze.load_cell(tmp_path)
    out = ~data["membership"][0]
    assert (data["sigma_rounds"][0][out][:, 25:] == 0).all()
    assert data["sigma_hat"][0, out, 0] == pytest.approx(np.full(8, 0.005), rel=0.01)
    assert data["oracle_sigma"][0, out] == pytest.approx(np.full(8, 0.005))
    assert data["collapsed_rounds"].tolist() == [0, 25] * 8
    summary = analyze.cell_summary(data)
    assert summary["collapsed_trajectories"] == 8 and summary["collapsed_rounds"] == 200
    assert summary["noise_level"]["auc"]["estimate"] > 0.95  # collapse is part of the observable channel


def test_a_round_missing_its_norm_without_collapse_is_an_error(tmp_path):
    combos = runner.build_cell("alzheimer", 3, "global-dp", multiple=4, stage="pilot")
    write_cell(tmp_path, combos, privacy="global-dp")
    folder = tmp_path / combos[0].run_name()
    path = folder / f"{combos[0].run_name()}.json"
    result = json.loads(path.read_text())
    del result["train_metrics"]["7"]["dp-released-update-norm"]
    path.write_text(json.dumps(result))
    with pytest.raises(KeyError):
        analyze.load_cell(tmp_path)


@pytest.mark.parametrize("multiple", [1, 16, 64])
def test_a_level_with_collapse_does_not_qualify(multiple):
    """Amendment 2026-10-19: more noise only collapses sooner, so the cell has no qualifying level."""
    assert analyze.pilot_decision(0.3, multiple, collapsed_trajectories=1) == ("no qualifying noise", None)
