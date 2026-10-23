"""Loading randomized-inclusion runs into attack-library arrays."""

from __future__ import annotations

import json

import numpy as np
import pytest

from results.cia_v2.randomized_inclusion import analysis, runner


def write_run(root, run, rounds=(0, 5, 10), released=(1.0, 2.0, 3.0, 4.0, 5.0)):
    folder = root / run.run_name()
    folder.mkdir(parents=True)
    pinned = runner.manifest(run, surrogates=False) | {"checkpoint_rounds": list(rounds)}
    (folder / "manifest.json").write_text(json.dumps(pinned))
    rows = [{"round": r, "candidate": c, "included": c in run.included, "test_loss": 1.0,
             "test_accuracy": 0.5 + 0.01 * run.replicate, "shadow_loss": 100 * run.replicate + r + c / 100,
             "shadow_accuracy": 0.9}
            for r in rounds for c in run.profile.candidates]
    (folder / "measurements.json").write_text(json.dumps(rows))
    metrics = {str(t + 1): {"dp-released-update-norm": value} for t, value in enumerate(released)}
    (folder / f"{run.run_name()}.json").write_text(json.dumps({"train_metrics": metrics}))
    (folder / "complete.json").write_text(json.dumps({"complete": True}))


@pytest.fixture
def audit(tmp_path):
    runs = runner.build_runs(config="gdp", privacy="global-dp", noise_ratio=0.001, replicates=4, design_seed=0)
    for run in runs:
        write_run(tmp_path, run)
    return tmp_path, runs


def test_load_audit_orders_candidates_and_replicates(audit):
    root, runs = audit
    data = analysis.load_audit(root, config="gdp")
    assert data.losses.shape == (len(runner.CANDIDATES), 4, 3)
    assert data.rounds == (0, 5, 10)
    c, j = 3, 2
    assert data.losses[c, j, 1] == pytest.approx(100 * j + 5 + runner.CANDIDATES[c] / 100)
    expected = np.array([[candidate in run.included for run in runs] for candidate in runner.CANDIDATES])
    np.testing.assert_array_equal(data.membership, expected)
    np.testing.assert_allclose(data.final_accuracy, [0.5, 0.51, 0.52, 0.53])
    np.testing.assert_allclose(data.released_norms[0], [1, 2, 3, 4, 5])


def test_load_audit_refuses_an_incomplete_design(audit):
    root, runs = audit
    (root / runs[1].run_name() / "complete.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        analysis.load_audit(root, config="gdp")


def test_load_audit_refuses_mixed_designs(audit, tmp_path_factory):
    root, _ = audit
    other = runner.build_runs(config="gdp", privacy="global-dp", noise_ratio=0.001, replicates=4, design_seed=1)
    write_run(root, other[0])
    with pytest.raises(ValueError, match="design"):
        analysis.load_audit(root, config="gdp")


def test_load_audit_reads_the_panel_from_the_manifests(tmp_path):
    runs = runner.build_runs(config="az", privacy="global-dp", noise_ratio=0.001, replicates=4, design_seed=0,
                             dataset="alzheimer")
    for run in runs:
        write_run(tmp_path, run)
    data = analysis.load_audit(tmp_path, config="az")
    assert data.candidates == runner.ALZHEIMER_CANDIDATES and data.top_decile == runner.ALZHEIMER_TOP_DECILE
    c, j = 3, 2
    assert data.losses[c, j, 1] == pytest.approx(100 * j + 5 + runner.ALZHEIMER_CANDIDATES[c] / 100)
    expected = np.array([[candidate in run.included for run in runs] for candidate in runner.ALZHEIMER_CANDIDATES])
    np.testing.assert_array_equal(data.membership, expected)
