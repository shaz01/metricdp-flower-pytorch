"""Tests for the CIFAR-100 CIA round-matched AUC analysis."""

from __future__ import annotations

import json

import pytest

from experiments.cia.scripts.cifar100_scaling_analysis import (
    _should_refuse_overwrite,
    build_summary,
    group_by_combo,
    main,
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
    # Seeds 42 and 43 have clean separation (IN scores higher than OUT); seed 44's IN/OUT
    # scores are reversed relative to the other two. A test that kept all 3 seeds concordant
    # would produce round_matched_auc == 1.0 -- the same value the pre-pooling, single-seed-only
    # code would have produced for seed 42 alone, so it wouldn't actually prove pooling across
    # seeds is happening. Reversing one seed makes the result (2/3) only reachable by combining
    # information from all 3 seeds' round-matched pairs.
    rounds = (1, 10, 20)
    in_rows = []
    out_rows = []
    for seed in (42, 43):
        for round_number in rounds:
            in_rows.append(
                _full_row(
                    partition="homogeneous", privacy="vanilla", seed=seed,
                    round_number=round_number, clean_loss=0.1, noisy_loss=0.1,
                )
            )
            out_rows.append(
                _full_row(
                    partition="homogeneous", privacy="vanilla", seed=seed,
                    round_number=round_number, clean_loss=0.9, noisy_loss=0.9,
                )
            )
    for round_number in rounds:
        # Seed 44: reversed -- OUT now has the lower loss (higher score) than IN.
        in_rows.append(
            _full_row(
                partition="homogeneous", privacy="vanilla", seed=44,
                round_number=round_number, clean_loss=0.9, noisy_loss=0.9,
            )
        )
        out_rows.append(
            _full_row(
                partition="homogeneous", privacy="vanilla", seed=44,
                round_number=round_number, clean_loss=0.1, noisy_loss=0.1,
            )
        )

    summary = build_summary(group_by_combo(in_rows), group_by_combo(out_rows))

    assert len(summary) == 1
    entry = summary[0]
    assert entry["partition_mode"] == "homogeneous"
    assert entry["privacy"] == "vanilla"
    # 3 seeds x 3 rounds = 9 round-matched pairs: 6 concordant (seeds 42, 43), 3 discordant
    # (seed 44) -> AUC 6/9 == 2/3. The old flat-grouped (non-seed-pooled) code could not produce
    # this specific fraction -- it structurally only ever saw one seed's worth of pairs.
    assert entry["target_clean_shadow_loss"]["round_matched_auc"] == pytest.approx(2 / 3)
    assert entry["target_clean_shadow_loss"]["pooled_auc"] == pytest.approx(2 / 3)


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
    # One seed's OUT rows land on different rounds than its IN rows. Both round lists still have
    # the same *length* (1 each), so round_matched_auc's zip(strict=True) does not raise -- the
    # ValueError instead comes from this file's own _validate_round_alignment, which checks round
    # *values*, not just counts. This test is the only coverage of that validator.
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


def test_should_refuse_overwrite_when_summary_empty_and_existing_file_nonempty(tmp_path) -> None:
    output_path = tmp_path / "cia_analysis.json"
    output_path.write_text(json.dumps([{"partition_mode": "homogeneous"}]), encoding="utf-8")

    assert _should_refuse_overwrite([], output_path) is True


def test_should_not_refuse_overwrite_when_summary_nonempty(tmp_path) -> None:
    output_path = tmp_path / "cia_analysis.json"
    output_path.write_text(json.dumps([{"partition_mode": "homogeneous"}]), encoding="utf-8")

    assert _should_refuse_overwrite([{"partition_mode": "non-iid"}], output_path) is False


def test_should_not_refuse_overwrite_when_no_existing_file(tmp_path) -> None:
    output_path = tmp_path / "cia_analysis.json"

    assert _should_refuse_overwrite([], output_path) is False


def test_should_not_refuse_overwrite_when_existing_file_already_empty(tmp_path) -> None:
    output_path = tmp_path / "cia_analysis.json"
    output_path.write_text("[]", encoding="utf-8")

    assert _should_refuse_overwrite([], output_path) is False


def test_main_refuses_to_overwrite_existing_nonempty_analysis_when_summary_empty(
    tmp_path, monkeypatch, capsys
) -> None:
    # Only seed 42's rows exist (as is the real committed state right now) -- every mechanism
    # is missing seeds 43/44's groups entirely, so build_summary returns [] for all of them.
    in_rows, out_rows = _rows_for_all_seeds(
        partition="homogeneous", privacy="vanilla", rounds=(1,), in_loss=0.1, out_loss=0.9,
    )
    in_rows = [row for row in in_rows if row["seed"] == 42]
    out_rows = [row for row in out_rows if row["seed"] == 42]

    results_dir = tmp_path
    (results_dir / "cia_in.json").write_text(json.dumps(in_rows), encoding="utf-8")
    (results_dir / "cia_out.json").write_text(json.dumps(out_rows), encoding="utf-8")
    existing_analysis = [{"partition_mode": "homogeneous", "privacy": "vanilla", "real": True}]
    analysis_path = results_dir / "cia_analysis.json"
    analysis_path.write_text(json.dumps(existing_analysis), encoding="utf-8")

    monkeypatch.setattr(
        "sys.argv", ["cifar100_scaling_analysis", "--results-dir", str(results_dir)]
    )

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code != 0
    assert json.loads(analysis_path.read_text(encoding="utf-8")) == existing_analysis
    warning = capsys.readouterr().out
    assert "refusing to overwrite" in warning


def test_main_writes_normally_when_no_existing_analysis_file(
    tmp_path, monkeypatch
) -> None:
    # Same partial (seed-42-only) data as above, but no pre-existing cia_analysis.json -- there
    # is nothing real to lose, so main() should write the (empty) summary normally.
    in_rows, out_rows = _rows_for_all_seeds(
        partition="homogeneous", privacy="vanilla", rounds=(1,), in_loss=0.1, out_loss=0.9,
    )
    in_rows = [row for row in in_rows if row["seed"] == 42]
    out_rows = [row for row in out_rows if row["seed"] == 42]

    results_dir = tmp_path
    (results_dir / "cia_in.json").write_text(json.dumps(in_rows), encoding="utf-8")
    (results_dir / "cia_out.json").write_text(json.dumps(out_rows), encoding="utf-8")
    analysis_path = results_dir / "cia_analysis.json"

    monkeypatch.setattr(
        "sys.argv", ["cifar100_scaling_analysis", "--results-dir", str(results_dir)]
    )

    main()

    assert json.loads(analysis_path.read_text(encoding="utf-8")) == []
