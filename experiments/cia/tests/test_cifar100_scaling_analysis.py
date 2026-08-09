"""Tests for the CIFAR-100 CIA round-matched AUC analysis."""

from __future__ import annotations

import pytest

from experiments.cia.scripts.cifar100_scaling_analysis import (
    group_by_combo,
    round_matched_auc,
)


def _row(*, partition: str, privacy: str, round_number: int, loss: float) -> dict:
    return {
        "partition_mode": partition,
        "privacy": privacy,
        "server_round": round_number,
        "target_clean_shadow_loss": loss,
    }


def test_group_by_combo_splits_on_partition_and_privacy() -> None:
    rows = [
        _row(partition="homogeneous", privacy="vanilla", round_number=1, loss=1.0),
        _row(partition="homogeneous", privacy="vanilla", round_number=10, loss=0.9),
        _row(partition="non-iid", privacy="vanilla", round_number=1, loss=1.1),
    ]

    groups = group_by_combo(rows)

    assert set(groups) == {("homogeneous", "vanilla"), ("non-iid", "vanilla")}
    assert [row["server_round"] for row in groups[("homogeneous", "vanilla")]] == [1, 10]


def test_group_by_combo_sorts_rows_by_round() -> None:
    rows = [
        _row(partition="homogeneous", privacy="vanilla", round_number=10, loss=0.9),
        _row(partition="homogeneous", privacy="vanilla", round_number=1, loss=1.0),
    ]

    groups = group_by_combo(rows)

    assert [row["server_round"] for row in groups[("homogeneous", "vanilla")]] == [1, 10]


def test_round_matched_auc_perfect_separation() -> None:
    # Higher shadow loss -> lower score (attack_scores negates), so a target that always
    # trains IN should show *lower* shadow loss than OUT at every matched round.
    in_rows = [
        _row(partition="homogeneous", privacy="vanilla", round_number=r, loss=0.1)
        for r in (1, 10, 20)
    ]
    out_rows = [
        _row(partition="homogeneous", privacy="vanilla", round_number=r, loss=0.9)
        for r in (1, 10, 20)
    ]

    result = round_matched_auc(in_rows, out_rows, "target_clean_shadow_loss")

    assert result == pytest.approx(1.0)


def test_round_matched_auc_requires_identical_rounds() -> None:
    in_rows = [_row(partition="homogeneous", privacy="vanilla", round_number=1, loss=0.1)]
    out_rows = [_row(partition="homogeneous", privacy="vanilla", round_number=10, loss=0.9)]

    with pytest.raises(ValueError, match="identical checkpoint rounds"):
        round_matched_auc(in_rows, out_rows, "target_clean_shadow_loss")
