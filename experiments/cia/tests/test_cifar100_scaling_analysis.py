"""Tests for the CIFAR-100 CIA round-matched AUC analysis."""

from __future__ import annotations

from experiments.cia.scripts.cifar100_scaling_analysis import (
    build_summary,
    group_by_combo,
)


def _row(
    *, partition: str, privacy: str, seed: int, round_number: int, loss: float
) -> dict:
    return {
        "partition_mode": partition,
        "privacy": privacy,
        "seed": seed,
        "server_round": round_number,
        "target_clean_shadow_loss": loss,
    }


def _full_row(
    *,
    partition: str,
    privacy: str,
    seed: int,
    round_number: int,
    clean_loss: float,
    noisy_loss: float,
) -> dict:
    """Like `_row`, but also carries `target_noisy_shadow_loss` for `build_summary`.

    `build_summary` scores both `SCORE_KEYS` per combo, so rows destined for it need both
    loss columns; the plain `_row` helper above is enough for the lower-level
    `group_by_combo` tests.
    """
    return {
        "partition_mode": partition,
        "privacy": privacy,
        "seed": seed,
        "server_round": round_number,
        "target_clean_shadow_loss": clean_loss,
        "target_noisy_shadow_loss": noisy_loss,
    }


def _rows_for_all_seeds(*, partition, privacy, rounds, in_loss, out_loss):
    """Build matched IN/OUT full-rows for every one of (42, 43, 44), same rounds each seed."""
    in_rows = [
        _full_row(
            partition=partition, privacy=privacy, seed=seed, round_number=r,
            clean_loss=in_loss, noisy_loss=in_loss,
        )
        for seed in (42, 43, 44)
        for r in rounds
    ]
    out_rows = [
        _full_row(
            partition=partition, privacy=privacy, seed=seed, round_number=r,
            clean_loss=out_loss, noisy_loss=out_loss,
        )
        for seed in (42, 43, 44)
        for r in rounds
    ]
    return in_rows, out_rows


def test_group_by_combo_splits_on_mechanism_and_seed() -> None:
    rows = [
        _row(partition="homogeneous", privacy="vanilla", seed=42, round_number=1, loss=1.0),
        _row(partition="homogeneous", privacy="vanilla", seed=42, round_number=10, loss=0.9),
        _row(partition="homogeneous", privacy="vanilla", seed=43, round_number=1, loss=1.1),
        _row(partition="non-iid", privacy="vanilla", seed=42, round_number=1, loss=1.2),
    ]

    groups = group_by_combo(rows)

    assert set(groups) == {
        ("homogeneous:vanilla", 42),
        ("homogeneous:vanilla", 43),
        ("non-iid:vanilla", 42),
    }
    assert [row["server_round"] for row in groups[("homogeneous:vanilla", 42)]] == [1, 10]


def test_group_by_combo_sorts_rows_by_round() -> None:
    rows = [
        _row(partition="homogeneous", privacy="vanilla", seed=42, round_number=10, loss=0.9),
        _row(partition="homogeneous", privacy="vanilla", seed=42, round_number=1, loss=1.0),
    ]

    groups = group_by_combo(rows)

    assert [row["server_round"] for row in groups[("homogeneous:vanilla", 42)]] == [1, 10]


def test_build_summary_pools_all_three_seeds() -> None:
    # Perfect separation across all 3 seeds -- IN always scores higher (lower loss) than OUT.
    in_rows, out_rows = _rows_for_all_seeds(
        partition="homogeneous", privacy="vanilla", rounds=(1, 10, 20),
        in_loss=0.1, out_loss=0.9,
    )

    summary = build_summary(group_by_combo(in_rows), group_by_combo(out_rows))

    assert len(summary) == 1
    entry = summary[0]
    assert entry["partition_mode"] == "homogeneous"
    assert entry["privacy"] == "vanilla"
    # 3 seeds x 3 rounds = 9 round-matched pairs, all concordant -> AUC 1.0.
    assert entry["target_clean_shadow_loss"]["round_matched_auc"] == 1.0
    assert entry["target_clean_shadow_loss"]["pooled_auc"] == 1.0


def test_build_summary_skips_mechanism_missing_from_one_side(capsys) -> None:
    in_rows, _unused = _rows_for_all_seeds(
        partition="non-iid", privacy="global-dp", rounds=(1,), in_loss=0.1, out_loss=0.9,
    )

    summary = build_summary(group_by_combo(in_rows), group_by_combo([]))

    assert summary == []
    warning = capsys.readouterr().out
    assert "non-iid:global-dp" in warning


def test_build_summary_skips_mechanism_missing_one_seed(capsys) -> None:
    # Present for seeds 42 and 43 on both sides, but seed 44 only has IN rows -- an entirely
    # missing seed-group (not just a round-count mismatch within a seed), which round_matched_auc
    # surfaces as a plain KeyError, not a ValueError.
    in_rows = [
        _full_row(
            partition="homogeneous", privacy="vanilla", seed=seed, round_number=1,
            clean_loss=0.1, noisy_loss=0.1,
        )
        for seed in (42, 43, 44)
    ]
    out_rows = [
        _full_row(
            partition="homogeneous", privacy="vanilla", seed=seed, round_number=1,
            clean_loss=0.9, noisy_loss=0.9,
        )
        for seed in (42, 43)
    ]

    summary = build_summary(group_by_combo(in_rows), group_by_combo(out_rows))

    assert summary == []
    warning = capsys.readouterr().out
    assert "homogeneous:vanilla" in warning


def test_build_summary_skips_mismatched_seed_but_keeps_healthy_mechanisms(capsys) -> None:
    healthy_in, healthy_out = _rows_for_all_seeds(
        partition="homogeneous", privacy="vanilla", rounds=(1, 10, 20),
        in_loss=0.1, out_loss=0.9,
    )
    # One seed's OUT rows land on different rounds than its IN rows -- round_matched_auc raises
    # ValueError (zip strict mismatch) for this mechanism, not KeyError.
    mismatched_in = [
        _full_row(
            partition="non-iid", privacy="metric-privacy", seed=seed, round_number=1,
            clean_loss=0.1, noisy_loss=0.1,
        )
        for seed in (42, 43, 44)
    ]
    mismatched_out = [
        _full_row(
            partition="non-iid", privacy="metric-privacy", seed=seed, round_number=10,
            clean_loss=0.9, noisy_loss=0.9,
        )
        for seed in (42, 43, 44)
    ]

    in_groups = group_by_combo(healthy_in + mismatched_in)
    out_groups = group_by_combo(healthy_out + mismatched_out)

    summary = build_summary(in_groups, out_groups)

    assert len(summary) == 1
    assert summary[0]["partition_mode"] == "homogeneous"
    warning = capsys.readouterr().out
    assert "non-iid:metric-privacy" in warning
