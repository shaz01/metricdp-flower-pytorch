"""Tests for the EuroSAT sweep script's combo-construction logic."""

from __future__ import annotations

import json
from pathlib import Path

from experiments.eurosat_scaling.sweep_eurosat_scaling import (
    CLIENT_COUNTS,
    build_combo,
    is_complete,
    result_path,
    run_name,
)


def test_run_name_is_deterministic_and_includes_key_parameters() -> None:
    name = run_name("homogeneous", "global-dp", "fedavg", 48, 100)

    assert name == run_name("homogeneous", "global-dp", "fedavg", 48, 100)
    assert "homogeneous" in name
    assert "global-dp" in name
    assert "fedavg" in name
    assert "n48" in name
    assert "r100" in name


def test_result_path_uses_output_dir_and_run_name(tmp_path, monkeypatch) -> None:
    import experiments.eurosat_scaling.sweep_eurosat_scaling as sweep

    monkeypatch.setattr(sweep, "OUTPUT_DIR", tmp_path)

    path = result_path("non-iid", "vanilla", "fedavg", 48, 100)

    assert path.parent == tmp_path
    assert path.name == f"{run_name('non-iid', 'vanilla', 'fedavg', 48, 100)}.json"


def test_client_counts_is_48_not_100() -> None:
    """EuroSAT has far fewer images than CIFAR-100 (21,600 vs 50,000 train) -- n=48
    keeps per-client data volume comparable, matching this plan's design rationale."""
    assert CLIENT_COUNTS == (48,)


def test_is_complete_false_for_missing_file(tmp_path) -> None:
    missing = tmp_path / "does-not-exist.json"

    assert is_complete(missing, expected_rounds=100) is False


def test_is_complete_false_when_evaluation_json_missing(tmp_path) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_text(
        json.dumps({"server_evaluate_metrics": {str(r): {} for r in range(1, 101)}}),
        encoding="utf-8",
    )
    # No run.evaluation.json written -- is_complete must require it too.

    assert is_complete(run_path, expected_rounds=100) is False


def test_is_complete_true_when_rounds_and_evaluation_both_present(tmp_path) -> None:
    run_path = tmp_path / "run.json"
    run_path.write_text(
        json.dumps({"server_evaluate_metrics": {str(r): {} for r in range(1, 101)}}),
        encoding="utf-8",
    )
    (tmp_path / "run.evaluation.json").write_text("{}", encoding="utf-8")

    assert is_complete(run_path, expected_rounds=100) is True


def test_build_combo_produces_valid_runner_args() -> None:
    """Constructs a real Combo/Hyperparams and calls the real runner_args() --
    this is what actually exercises the Hyperparams field set against this
    branch's real matrix module, rather than only the sweep script's own
    string-formatting helpers."""
    combo = build_combo("homogeneous", "global-dp", "fedavg", 48, 100)

    args = combo.runner_args(
        output_dir=Path("/tmp/eurosat-test"), max_parallel_clients=16, client_cpus=1.0
    )

    assert "--data-module" in args
    assert "experiments.reproduce.dataset.eurosat:create_data_module" in args
    assert "--model-module" in args
    assert "experiments.reproduce.eurosat_cnn:create_model" in args
    assert "--num-clients" in args
    assert "48" in args
