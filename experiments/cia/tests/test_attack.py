"""Tests for the CIA relative-difference attack score (Section 7.4.1)."""

from __future__ import annotations

import pytest

from experiments.cia.result import CiaResult, make_cia_result, relative_difference
from experiments.cia.runner import build_cia_combos


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
    combo = build_cia_combos(
        privacy_modes=("vanilla",), aggregations=("fedavg",)
    )[0]
    result = make_cia_result(
        combo=combo,
        server_round=1,
        aggregated_test_loss=1.032,
        target_shadow_loss=1.182,
        shadow_fraction=0.10,
        shadow_size=150,
    )
    assert isinstance(result, CiaResult)
    assert result.privacy == "vanilla"
    assert result.aggregation == "fedavg"
    assert result.server_round == 1
    assert result.num_clients == 3
    assert result.shadow_fraction == 0.10
    assert result.shadow_size == 150
    assert result.difference_pct == pytest.approx(12.719, abs=0.1)
