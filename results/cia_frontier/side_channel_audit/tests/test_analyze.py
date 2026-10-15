"""Synthetic results tree -> analyze.py: trial construction, orientation and gate decisions."""

from __future__ import annotations

import json

import numpy as np
import pytest

from results.cia_frontier.side_channel_audit import analyze

ROUNDS, CLIENTS, SEEDS = 6, 3, (42, 43, 44)
ALZHEIMER = "results.cia_frontier.side_channel_audit.data:create_data_module"


def _write(root, privacy, seed, out_target, rng, *, separable):
    """IN (target present): sigma 0.9 vs 1.0 and increments -0.5 vs 0 when ``separable``."""
    name = f"{privacy}-{seed}-{out_target}"
    folder = root / privacy / name
    folder.mkdir(parents=True)
    present = lambda t: out_target is None  # the target t is IN only in the IN trajectory
    shift = 0.5 if separable else 0.0
    manifest = {"alpha": 0.3, "seed": seed, "privacy": privacy, "noise_ratio": 0.01, "clients": CLIENTS,
                "out_target": out_target, "targets": [out_target] if out_target is not None else list(range(CLIENTS)),
                "rounds": ROUNDS, "pilot": False, "schema": 1, "run_name": name, "diagnostics": True,
                "data_module": ALZHEIMER}
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "complete.json").write_text("{}")
    rows = []
    for target in manifest["targets"]:
        increments = rng.normal(-shift if present(target) else 0.0, 0.2, ROUNDS)
        losses = 2.0 + np.cumsum(increments)
        for r in range(1, ROUNDS + 1):
            row = {"round": r, "target": target, "aggregate_loss": 1.0, "shadow_size": 10}
            for view in ("clean", "noisy"):
                row.update({f"{view}_loss": losses[r - 1], f"{view}_increment": increments[r - 1],
                            f"{view}_loss_before": losses[r - 1] - increments[r - 1],
                            f"{view}_grad_norm_start": 1.0})
            rows.append(row)
    (folder / "measurements.json").write_text(json.dumps(rows))
    sigma = (0.9 if (separable and out_target is None) else 1.0) * (1 + 0.01 * rng.normal(size=ROUNDS))
    (folder / "releases.json").write_text(json.dumps(
        [{"round": r, "update_norm": float(sigma[r - 1] * 10.0), "parameter_count": 100}
         for r in range(1, ROUNDS + 1)]))
    key = "metric-dp-noise-stdv" if privacy == "metric-privacy" else "global-dp-noise-stdv"
    run = {"train_metrics": {str(r): {key: float(sigma[r - 1])} for r in range(1, ROUNDS + 1)},
           "server_evaluate_metrics": {str(ROUNDS): {"accuracy": 0.8}}}
    (folder / f"{name}.json").write_text(json.dumps(run))
    (folder / "partitions.json").write_text(json.dumps({"clients": [{"records": n} for n in (300, 900, 500)]}))


def _tree(tmp_path, separable):
    rng = np.random.default_rng(0)
    for privacy in ("global-dp", "metric-privacy"):
        for seed in SEEDS:
            for out_target in (None, *range(CLIENTS)):
                _write(tmp_path, privacy, seed, out_target, rng, separable=separable)
    return tmp_path


@pytest.fixture(autouse=True)
def _few_resamples(monkeypatch):
    monkeypatch.setattr(analyze, "RESAMPLES", 200)


def test_separable_tree_passes_both_gate_parts(tmp_path) -> None:
    report = analyze.analyze(_tree(tmp_path, separable=True))
    assert report["gate"]["G1a"]["decision"] == "pass"
    assert report["gate"]["G1a"]["value"] > 0.9
    cell = {(c["dataset"], c["privacy"]): c for c in report["cells"]}[("alzheimer", "metric-privacy")]
    attacks = cell["attacks"]["clean/all"]
    assert attacks["increment"]["auc"]["value"] > 0.9
    assert attacks["oracle_noise_level"]["auc"]["value"] > 0.9
    assert cell["in_accuracy"] == [0.8, 0.8, 0.8]


def test_null_tree_does_not_pass(tmp_path) -> None:
    report = analyze.analyze(_tree(tmp_path, separable=False))
    assert report["gate"]["G1a"]["decision"] != "pass"
    assert abs(report["gate"]["G1a"]["value"] - 0.5) < 0.25


def test_top_decile_is_the_largest_client_and_references_exclude_the_observed_seed(tmp_path) -> None:
    assert analyze.top_decile([300, 900, 500]) == {1}
    assert analyze.top_decile(list(range(20))) == {19, 18}
    cells = analyze.load(_tree(tmp_path, separable=True))
    cell = cells[("alzheimer", 3, "metric-privacy")]
    clusters = analyze.clusters(cell, subgroup="top_decile")
    assert [(t, s) for t, s, _ in clusters] == [(1, 42), (1, 43), (1, 44)]
    # Removing the observed seed's own OUT trajectory from the cell must not change its score.
    before = analyze.trial_scores(cell, 42, 0, "in")["noise_level"]
    cell["out"][(42, 0)] = None
    np.testing.assert_allclose(analyze.trial_scores(cell, 42, 0, "in")["noise_level"], before)


def test_incomplete_or_unbalanced_designs_are_rejected(tmp_path) -> None:
    root = _tree(tmp_path, separable=True)
    victim = next(root.rglob("complete.json"))
    victim.unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        analyze.load(root)


def test_gate_decision_rule() -> None:
    assert analyze.gate_decision(0.56, 0.7, 0.55) == "pass"
    assert analyze.gate_decision(0.4, 0.55, 0.55) == "fail"
    assert analyze.gate_decision(0.5, 0.6, 0.55) == "inconclusive"


def test_side_channel_attribution_rule() -> None:
    assert analyze.side_channel_attribution(0.6, 0.8, 0.7) == "metric-privacy noise level"
    assert analyze.side_channel_attribution(0.5, 0.8, 0.7) == "update norm (both mechanisms)"
    assert analyze.side_channel_attribution(0.6, 0.7, 0.8) == "update norm (both mechanisms)"


def test_report_carries_the_attribution_inputs(tmp_path) -> None:
    g1a = analyze.analyze(_tree(tmp_path, separable=True))["gate"]["G1a"]
    assert {"oracle_metric_privacy_auc", "blind_global_dp_auc", "attribution"} <= set(g1a)
