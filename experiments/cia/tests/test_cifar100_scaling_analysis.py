"""Tests for the CIFAR-100 CIA round-matched AUC analysis."""

from __future__ import annotations

import pytest

from experiments.cia.scripts.cifar100_scaling_analysis import (
    build_summary,
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


def _full_row(
    *,
    partition: str,
    privacy: str,
    round_number: int,
    clean_loss: float,
    noisy_loss: float,
) -> dict:
    """Like `_row`, but also carries `target_noisy_shadow_loss` for `build_summary`.

    `build_summary` scores both `SCORE_KEYS` per combo, so rows destined for it need both
    loss columns; the plain `_row` helper above (which only sets the clean-loss key) is enough
    for the lower-level `round_matched_auc`/`group_by_combo` tests.
    """
    return {
        "partition_mode": partition,
        "privacy": privacy,
        "server_round": round_number,
        "target_clean_shadow_loss": clean_loss,
        "target_noisy_shadow_loss": noisy_loss,
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


def test_build_summary_skips_mismatched_combo_but_keeps_healthy_ones(capsys) -> None:
    # A designed-for outcome, not a bug: run_attack continues past training failures, so one
    # combo can end up with a partial/interrupted trajectory whose IN/OUT rounds don't match
    # while the rest of the sweep is healthy.
    healthy_in = [
        _full_row(
            partition="homogeneous", privacy="vanilla", round_number=r,
            clean_loss=0.1, noisy_loss=0.2,
        )
        for r in (1, 10, 20)
    ]
    healthy_out = [
        _full_row(
            partition="homogeneous", privacy="vanilla", round_number=r,
            clean_loss=0.9, noisy_loss=0.8,
        )
        for r in (1, 10, 20)
    ]
    mismatched_in = [
        _full_row(
            partition="non-iid", privacy="metric-privacy", round_number=1,
            clean_loss=0.1, noisy_loss=0.2,
        )
    ]
    mismatched_out = [
        _full_row(
            partition="non-iid", privacy="metric-privacy", round_number=10,
            clean_loss=0.9, noisy_loss=0.8,
        )
    ]

    in_groups = group_by_combo(healthy_in + mismatched_in)
    out_groups = group_by_combo(healthy_out + mismatched_out)

    summary = build_summary(in_groups, out_groups)

    assert len(summary) == 1
    assert summary[0]["partition_mode"] == "homogeneous"
    assert summary[0]["privacy"] == "vanilla"
    warning = capsys.readouterr().out
    assert "non-iid" in warning
    assert "metric-privacy" in warning


def test_build_summary_skips_combo_missing_from_one_side(capsys) -> None:
    in_only_rows = [
        _full_row(
            partition="non-iid", privacy="global-dp", round_number=1,
            clean_loss=0.1, noisy_loss=0.2,
        )
    ]

    in_groups = group_by_combo(in_only_rows)
    out_groups: dict = {}

    summary = build_summary(in_groups, out_groups)

    assert summary == []
    warning = capsys.readouterr().out
    assert "non-iid" in warning
    assert "global-dp" in warning
