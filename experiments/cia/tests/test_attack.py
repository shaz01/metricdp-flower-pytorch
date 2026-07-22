"""Tests for the CIA relative-difference attack score (Section 7.4.1)."""

from __future__ import annotations

import pytest

from experiments.cia.attack import CiaResult, make_cia_result, relative_difference


def test_relative_difference_exact_values() -> None:
    assert relative_difference(aggregated_loss=80.0, target_loss=100.0) == pytest.approx(20.0)
    assert relative_difference(aggregated_loss=100.0, target_loss=100.0) == pytest.approx(0.0)


def test_relative_difference_matches_paper_table_10_fedavg_vanilla() -> None:
    # Table 10, FedAvg, Vanilla FL: aggregated=1.032, target=1.182, paper reports 12.719%.
    result = relative_difference(aggregated_loss=1.032, target_loss=1.182)
    assert result == pytest.approx(12.719, abs=0.1)


def test_relative_difference_rejects_zero_target_loss() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        relative_difference(aggregated_loss=1.0, target_loss=0.0)


def test_make_cia_result_computes_difference_pct() -> None:
    result = make_cia_result(
        privacy="vanilla",
        aggregation="fedavg",
        aggregated_test_loss=1.032,
        target_shadow_loss=1.182,
    )
    assert isinstance(result, CiaResult)
    assert result.privacy == "vanilla"
    assert result.aggregation == "fedavg"
    assert result.difference_pct == pytest.approx(12.719, abs=0.1)
