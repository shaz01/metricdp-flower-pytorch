"""Tests for the CIFAR-10 removal-adjacency CIA scoring."""

from __future__ import annotations

import json

import pytest

from experiments.cia.scripts import cifar10_remove_analysis as analysis


def _row(server_round: int, clean: float, noisy: float, aggregated: float = 2.0) -> dict:
    return {
        "run_name": "run",
        "seed": 42,
        "partition_mode": "non-iid",
        "num_clients": 8,
        "privacy": "vanilla",
        "aggregation": "fedavg",
        "noise_multiplier": 0.0182,
        "server_round": server_round,
        "aggregated_test_loss": aggregated,
        "target_clean_shadow_loss": clean,
        "target_noisy_shadow_loss": noisy,
        "shadow_fraction": 0.1,
        "shadow_size": 100,
        "clean_difference_pct": (clean - aggregated) / clean * 100,
        "noisy_difference_pct": (noisy - aggregated) / noisy * 100,
    }


def _write_chunk(root, clients: int, adjacency: str, privacy: str, rows: list[dict]) -> None:
    path = root / f"clients-{clients}" / adjacency / privacy / "runs"
    path.mkdir(parents=True)
    (path / "cia.json").write_text(json.dumps(rows), encoding="utf-8")


def test_perfect_separation_scores_auc_one() -> None:
    """Every IN round having a lower shadow loss is a perfect attack."""
    in_rows = [_row(r, clean=0.1, noisy=0.2) for r in range(1, 21)]
    out_rows = [_row(r, clean=1.0, noisy=1.2) for r in range(1, 21)]

    entry = analysis.multi_round_entry(in_rows, out_rows)

    assert entry["clean"]["pooled_auc"] == 1.0
    assert entry["clean"]["round_matched_auc"] == 1.0
    assert entry["noisy"]["pooled_auc"] == 1.0


def test_identical_trajectories_score_auc_half() -> None:
    """No membership signal must land on the 0.5 chance line, not above it."""
    rows_in = [_row(r, clean=0.5, noisy=0.6) for r in range(1, 21)]
    rows_out = [_row(r, clean=0.5, noisy=0.6) for r in range(1, 21)]

    entry = analysis.multi_round_entry(rows_in, rows_out)

    assert entry["clean"]["pooled_auc"] == 0.5
    assert entry["clean"]["round_matched_auc"] == 0.5


def test_round_matched_auc_rejects_mismatched_checkpoints() -> None:
    in_rows = [_row(r, clean=0.1, noisy=0.2) for r in (1, 2, 3)]
    out_rows = [_row(r, clean=1.0, noisy=1.2) for r in (1, 2, 4)]

    with pytest.raises(ValueError):
        analysis.round_matched_auc(in_rows, out_rows, "target_clean_shadow_loss")


def test_first_round_entry_matches_the_papers_relative_difference() -> None:
    """(target - aggregated) / target * 100, the Tables 10-12 formula."""
    rows = [_row(1, clean=1.182, noisy=1.5, aggregated=1.032)] + [
        _row(r, clean=0.5, noisy=0.6) for r in range(2, 21)
    ]

    entry = analysis.first_round_entry(rows)

    assert entry["aggregated_test_loss"] == pytest.approx(1.032)
    assert entry["target_clean_shadow_loss"] == pytest.approx(1.182)
    assert entry["clean_difference_pct"] == pytest.approx(12.69, abs=0.01)


def test_bootstrap_interval_brackets_the_point_estimate() -> None:
    in_rows = [_row(r, clean=0.1 + 0.01 * r, noisy=0.2) for r in range(1, 21)]
    out_rows = [_row(r, clean=0.6 + 0.01 * r, noisy=0.9) for r in range(1, 21)]

    scores = analysis.multi_round_entry(in_rows, out_rows)["clean"]
    low, high = scores["pooled_auc_ci95"]

    assert low <= scores["pooled_auc"] <= high


def test_build_summary_reads_the_chunk_layout(tmp_path) -> None:
    in_rows = [_row(r, clean=0.1, noisy=0.2) for r in range(1, 21)]
    out_rows = [_row(r, clean=1.0, noisy=1.2) for r in range(1, 21)]
    _write_chunk(tmp_path, 8, "in-remove", "vanilla", in_rows)
    _write_chunk(tmp_path, 8, "out-remove", "vanilla", out_rows)

    assert analysis.discover_client_counts(tmp_path) == [8]
    summary = analysis.build_summary(tmp_path, [8])

    assert len(summary) == 1
    assert summary[0]["privacy"] == "vanilla"
    assert summary[0]["multi_round"]["clean"]["pooled_auc"] == 1.0


def test_build_summary_skips_a_combo_missing_its_out_side(tmp_path, capsys) -> None:
    """A half-finished pair must warn and be skipped, not abort the analysis."""
    _write_chunk(tmp_path, 8, "in-remove", "vanilla", [_row(1, 0.1, 0.2)])

    summary = analysis.build_summary(tmp_path, [8])

    assert summary == []
    assert "skipping" in capsys.readouterr().out
