"""P1 analysis on synthetic trajectories (no runs, no GPU)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from results.cia_v3.paper_setting import analyze


def synthetic(gap, pairs=20, seed=0, noise=0.05):
    rng = np.random.default_rng(seed)
    base = np.linspace(1.4, 0.3, 20)
    losses = np.empty((1, 2 * pairs, 20))
    for p in range(pairs):
        world_offset = rng.normal(0, noise, size=2)
        losses[0, 2 * p] = base - gap + world_offset[0] + rng.normal(0, noise, 20)
        losses[0, 2 * p + 1] = base + world_offset[1] + rng.normal(0, noise, 20)
    membership = np.tile([True, False], pairs)[None]
    return losses, membership


def test_paper_statistic_is_the_pooled_auc_per_pair():
    losses, membership = synthetic(gap=0.0, pairs=2)
    losses[0, 0] = np.arange(20) / 100.0        # IN: losses 0.00..0.19
    losses[0, 1] = np.arange(20) / 100.0 + 0.1  # OUT: losses 0.10..0.29
    value = analyze.paper_statistic(losses, membership)[0]
    expected = np.mean([[(a < b) + 0.5 * (a == b) for b in losses[0, 1]] for a in losses[0, 0]])
    assert value == pytest.approx(expected)


def test_no_signal_gives_chance_and_a_wide_paper_statistic():
    losses, membership = synthetic(gap=0.0, noise=0.05, seed=1)
    summary = analyze.summarize(losses, losses, membership)
    assert 0.3 < summary["calibrated"]["estimate"] < 0.7
    assert summary["calibrated"]["low"] < 0.5
    stat = summary["paper_statistic"]
    assert stat["p05"] < 0.5 < stat["p95"]


def test_strong_signal_is_detected():
    losses, membership = synthetic(gap=0.2, noise=0.05, seed=2)
    summary = analyze.summarize(losses, losses, membership)
    assert summary["calibrated"]["low"] > 0.5
    assert summary["paper_statistic"]["median"] > 0.5


def test_decision_rule():
    def config(low, p05, p95):
        return {"calibrated": {"low": low}, "paper_statistic": {"p05": p05, "p95": p95}}

    assert analyze.decide({"g": config(0.6, 0.6, 0.9), "m": config(0.55, 0.7, 0.9)}, "g", "m") == [
        "the published protection claim fails in its own setting"]
    assert analyze.decide({"g": config(0.6, 0.3, 0.9), "m": config(0.45, 0.6, 0.9)}, "g", "m") == [
        "Table 13 is a single draw from a wide distribution"]
    assert analyze.decide({"g": config(0.6, 0.3, 0.9), "m": config(0.55, 0.6, 0.9)}, "g", "m") == [
        "the published protection claim fails in its own setting", "Table 13 is a single draw from a wide distribution"]
    assert analyze.decide({"g": config(0.4, 0.6, 0.9), "m": config(0.55, 0.6, 0.9)}, "g", "m") == [
        "consistent with the paper"]


def _write_run(root, config, pair, world, losses, accuracy=0.9):
    folder = root / f"paper-{config}-d1-pair{pair:02d}-{world}"
    folder.mkdir(parents=True)
    run_name = folder.name + "__x"
    manifest = {"config": config, "pair": pair, "world": world, "design_seed": 1, "pairs": 2,
                "noise_multiplier": 0.01, "privacy": "global-dp", "run_name": run_name}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "complete.json").write_text("{}")
    rows = [{"round": r, "shadow_loss": float(losses[r]), "noisy_shadow_loss": float(losses[r]) + 0.01}
            for r in range(21)]
    (folder / "measurements.json").write_text(json.dumps(rows))
    metrics = {str(r): {"eval_acc": accuracy, "per-client-eval_acc": [accuracy, accuracy - 0.1]} for r in range(1, 21)}
    (folder / f"{run_name}.json").write_text(json.dumps({"client_evaluate_metrics": metrics}))


def test_load_config_orders_worlds_and_reads_rounds_1_to_20(tmp_path):
    for pair in (1, 0):
        for world in ("out", "in"):
            _write_run(tmp_path, "gdp", pair, world, np.arange(21) + 100 * pair + (world == "out") * 10.0)
    noisy, clean, membership, worlds = analyze.load_config(tmp_path, "gdp")
    assert noisy.shape == clean.shape == (1, 4, 20)
    assert membership.tolist() == [[True, False, True, False]]
    assert clean[0, 0, 0] == 1.0 and clean[0, 1, 0] == 11.0 and clean[0, 2, 0] == 101.0
    assert noisy[0, 0, 0] == pytest.approx(1.01)
    accuracy, lowest = analyze.table13_accuracy(worlds)
    assert accuracy.tolist() == [0.9] * 4 and lowest == pytest.approx([0.8] * 4)


def test_load_config_refuses_a_missing_world(tmp_path):
    _write_run(tmp_path, "gdp", 0, "in", np.arange(21.0))
    with pytest.raises(ValueError, match="lack a world"):
        analyze.load_config(tmp_path, "gdp")


def test_reproduces_m2_vanilla_auc():
    root = analyze.REPO / "results/cia_v2/frontier/results"
    if not root.exists():
        pytest.skip("M2 runs not checked out")
    assert round(analyze.reproduce_m2(root), 3) == analyze.M2_VANILLA_AUC
