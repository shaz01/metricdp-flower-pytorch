"""Tests for the shared dataset-agnostic CIA stage scorer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.cia.result import CiaResult
from experiments.cia.scripts import score_stage


def _result(run_name: str, server_round: int, clean_loss: float) -> CiaResult:
    return CiaResult(
        run_name=run_name,
        seed=42,
        partition_mode="homogeneous",
        num_clients=8,
        privacy="global-dp",
        aggregation="fedavg",
        noise_multiplier=0.01,
        server_round=server_round,
        aggregated_test_loss=1.0,
        target_clean_shadow_loss=clean_loss,
        target_noisy_shadow_loss=clean_loss,
        shadow_fraction=0.1,
        shadow_size=10,
        clean_difference_pct=0.0,
        noisy_difference_pct=0.0,
    )


def test_round_matched_clean_shadow_auc_perfect_separation() -> None:
    # IN's loss is always lower (i.e. "more confident"/more revealing) than OUT's.
    results = [
        _result("ds-in-remove", 1, 0.1),
        _result("ds-in-remove", 2, 0.2),
        _result("ds-out-remove", 1, 0.9),
        _result("ds-out-remove", 2, 0.8),
    ]
    assert score_stage.round_matched_clean_shadow_auc(results) == pytest.approx(1.0)


def test_round_matched_clean_shadow_auc_allows_direction_reversal() -> None:
    # IN's loss is always higher than OUT's -- directional AUC is 0.0, but the
    # attacker is allowed to reverse direction, so the effective AUC is 1.0.
    results = [
        _result("ds-in-remove", 1, 0.9),
        _result("ds-out-remove", 1, 0.1),
    ]
    assert score_stage.round_matched_clean_shadow_auc(results) == pytest.approx(1.0)


def test_round_matched_clean_shadow_auc_chance_level() -> None:
    results = [
        _result("ds-in-remove", 1, 0.5),
        _result("ds-in-remove", 2, 0.5),
        _result("ds-out-remove", 1, 0.5),
        _result("ds-out-remove", 2, 0.5),
    ]
    assert score_stage.round_matched_clean_shadow_auc(results) == pytest.approx(0.5)


def test_stage_accuracy_averages_final_round_across_runs(tmp_path: Path) -> None:
    for name, accuracy in (("ds-in-remove", 0.80), ("ds-out-remove", 0.90)):
        payload = {
            "server_evaluate_metrics": {
                "1": {"accuracy": 0.5},
                "2": {"accuracy": accuracy},
            }
        }
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))

    accuracy = score_stage.stage_accuracy(
        tmp_path, run_names=("ds-in-remove", "ds-out-remove")
    )
    assert accuracy == pytest.approx(0.85)


def test_score_stage_combines_auc_and_accuracy(tmp_path: Path) -> None:
    results = [
        _result("ds-in-remove", 1, 0.1),
        _result("ds-out-remove", 1, 0.9),
    ]
    for name, accuracy in (("ds-in-remove", 0.80), ("ds-out-remove", 0.90)):
        payload = {"server_evaluate_metrics": {"1": {"accuracy": accuracy}}}
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))

    score = score_stage.score_stage(results, tmp_path)
    assert score.auc == pytest.approx(1.0)
    assert score.accuracy == pytest.approx(0.85)
