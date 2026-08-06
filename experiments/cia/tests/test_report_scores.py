"""Score-construction helpers used by the Alzheimer CIA report."""

from __future__ import annotations

import pytest

from experiments.cia.reports.build_alzheimer_cia_report import (
    attack_scores,
    auc,
    discordant_breakdown,
    pooled_auc,
    round_matched_auc,
)


def _rows(losses: list[float]) -> list[dict]:
    return [
        {
            "server_round": index,
            "target_clean_shadow_loss": loss,
            "aggregated_test_loss": 1.0,
        }
        for index, loss in enumerate(losses, start=1)
    ]


def _groups(losses: list[float]) -> dict[tuple[str, int], list[dict]]:
    return {("m", seed): _rows(losses) for seed in (42, 43, 44)}


def test_attack_scores_negate_losses() -> None:
    assert attack_scores(_rows([0.5, 2.0]), "target_clean_shadow_loss").tolist() == [
        -0.5,
        -2.0,
    ]


def test_auc_counts_ties_as_half() -> None:
    scores = attack_scores(_rows([1.0]), "target_clean_shadow_loss")
    assert auc(scores, scores) == pytest.approx(0.5)


def test_round_matched_auc_ignores_cross_round_pairs() -> None:
    """IN wins at every matched round even though its early loss is the worst.

    The pooled statistic is dragged down by comparing IN round 1 against OUT
    round 3; the round-matched statistic is not.
    """
    in_groups = _groups([1.0, 0.5, 0.1])
    out_groups = _groups([1.1, 0.9, 0.8])

    assert round_matched_auc(in_groups, out_groups, "m", "target_clean_shadow_loss") == 1.0
    assert pooled_auc(in_groups, out_groups, "m", "target_clean_shadow_loss") < 1.0


def test_discordant_breakdown_isolates_the_round_confound() -> None:
    in_groups = _groups([1.0, 0.5, 0.1])
    out_groups = _groups([1.1, 0.9, 0.8])

    overall, same_round, in_earlier = discordant_breakdown(
        in_groups, out_groups, "m", "target_clean_shadow_loss"
    )

    assert same_round == 0.0
    assert in_earlier > 0.0
    assert 0.0 < overall < in_earlier


def test_round_matched_auc_requires_equal_round_counts() -> None:
    with pytest.raises(ValueError):
        round_matched_auc(
            _groups([1.0, 0.5]),
            _groups([1.0]),
            "m",
            "target_clean_shadow_loss",
        )
