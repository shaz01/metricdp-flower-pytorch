"""P3 analysis on synthetic arrays (no runs, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from research.cia_roadmap_v3.reanalysis import reanalyze as rz
from results.cia_v2.randomized_inclusion.runner import design_matrix
from results.cia_v3.robustness import analyze


def test_every_round_statistic_matches_the_window_statistic_at_window_one(monkeypatch):
    rng = np.random.default_rng(0)
    losses = rng.normal(size=(4, 6, 51))
    schedules = rng.random((4, 6, 50)) < 0.25
    monkeypatch.setattr(rz, "WINDOW", 1)
    assert analyze.every_round_statistic(losses, schedules, 0.25) == pytest.approx(
        rz.schedule_statistic(losses, schedules, 0.25))


def test_every_round_statistic_finds_rounds_where_the_client_trained():
    rng = np.random.default_rng(1)
    schedules = rng.random((8, 16, 50)) < 0.25
    losses = np.zeros((8, 16, 51))
    losses[:, :, 1:] = -np.cumsum(schedules * 0.1, axis=2)  # the loss drops 0.1 in each sampled round
    z = analyze.every_round_statistic(losses, schedules, 0.25)
    shuffled = rng.permuted(schedules, axis=2)
    assert z.mean() > 3 * abs(analyze.every_round_statistic(losses, shuffled, 0.25).mean())


def test_in_count_designs_are_complementary_blocks():
    eight, twenty_four = (design_matrix(32, design_seed=2, in_count=k) for k in (8, 24))
    assert np.array_equal(eight, ~twenty_four)
    for block in analyze.blocks_of(32):
        assert (eight[:, block].sum(axis=1) == 1).all()  # every candidate IN exactly once per block


def scored_design(in_count, gap, seed):
    rng = np.random.default_rng(seed)
    member = design_matrix(32, design_seed=2, in_count=in_count)
    return rz.Scored(rng.normal(size=member.shape) + gap * member, member, np.full(32, 0.7))


def test_in_count_difference_detects_a_gap_and_not_its_absence():
    same = analyze.in_count_difference(scored_design(8, 1.0, 0), scored_design(24, 1.0, 1))
    assert same.low < 0 < same.high
    apart = analyze.in_count_difference(scored_design(8, 2.0, 2), scored_design(24, 0.5, 3))
    assert apart.low > 0.2


def test_block_jackknife_of_a_constant_has_no_width():
    interval = analyze.block_jackknife(lambda kept: 0.5, 32)
    assert (interval.estimate, interval.low, interval.high) == (0.5, 0.5, 0.5)
    with pytest.raises(ValueError):
        analyze.blocks_of(30)


def test_c1_outcome_and_g_p3a():
    gate = lambda e, lo, hi: {"estimate": e, "low": lo, "high": hi}  # noqa: E731
    assert analyze.c1_outcome(gate(-0.06, -0.10, -0.02)) == "pass"
    assert analyze.c1_outcome(gate(-0.04, -0.08, -0.01)) == "fail"
    assert analyze.c1_outcome(gate(-0.06, -0.12, 0.01)) == "fail"
    assert analyze.c1_outcome(gate(float("nan"), float("nan"), float("nan"))) == "not testable"
    assert analyze.g_p3a(["pass", "not testable"]) == "pass"
    assert analyze.g_p3a(["fail", "not testable"]) == "fail"
    assert analyze.g_p3a(["not testable", "not testable"]) == "not testable"


def test_g_p3b_needs_the_whole_interval_inside():
    assert analyze.g_p3b({"low": -0.03, "high": 0.03}) == "pass"
    assert analyze.g_p3b({"low": -0.01, "high": 0.04}).startswith("escalate")
    assert analyze.g_p3b({"low": -0.04, "high": 0.01}).startswith("escalate")


def test_follow_up_decision():
    assert analyze.follow_up_decision({"low": -0.02, "high": 0.025}).startswith("no IN-count interference")
    assert analyze.follow_up_decision({"low": 0.031, "high": 0.08}).startswith("interference")
    assert analyze.follow_up_decision({"low": -0.09, "high": -0.035}).startswith("interference")
    assert analyze.follow_up_decision({"low": 0.01, "high": 0.07}) == "inconclusive at this power"


def write_runs(root, config, replicates, total, *, gap=0.3, seed=0, flip=None):
    """Fake runner output for the IN-count design (8 IN); ``flip`` = (candidate row, replicate)."""
    import json
    from results.cia_v2.randomized_inclusion.runner import CANDIDATES, TOP_DECILE
    rng = np.random.default_rng(seed)
    design = design_matrix(total, design_seed=2, in_count=8)
    rounds = list(range(0, 51, 5))
    for j in replicates:
        included = [c for c, member in zip(CANDIDATES, design[:, j]) if member]
        if flip is not None and flip[1] == j:
            candidate = CANDIDATES[flip[0]]
            included = [c for c in included if c != candidate] if candidate in included else included + [candidate]
        name = f"ri-{config}-d2-rep{j:02d}"
        folder = root / name
        folder.mkdir(parents=True)
        manifest = {"config": config, "replicate": j, "run_name": name, "design_seed": 2, "replicates": total,
                    "checkpoint_rounds": rounds, "candidates": list(CANDIDATES), "top_decile": list(TOP_DECILE),
                    "included": included, "in_count": 8}
        (folder / "manifest.json").write_text(json.dumps(manifest))
        rows = [{"round": t, "candidate": c, "test_accuracy": 0.7,
                 "shadow_loss": 2.0 - t / 50 - gap * (c in included) + rng.normal(0, 0.1)}
                for t in rounds for c in CANDIDATES]
        (folder / "measurements.json").write_text(json.dumps(rows))
        (folder / f"{name}.json").write_text(json.dumps({"train_metrics": {"1": {}},
                                                        "client_evaluate_metrics": {"50": {"per-client-eval_acc": [0.5]}}}))
        (folder / "complete.json").write_text("{}")


def test_extended_audit_joins_the_two_halves_of_the_design(tmp_path):
    write_runs(tmp_path, "base", range(32), 32)
    write_runs(tmp_path, "more", range(32, 64), 64, seed=1)
    audit = analyze.load_extended(tmp_path, "base", "more")
    assert audit.losses.shape == (32, 64, 11)
    assert np.array_equal(audit.membership, design_matrix(64, design_seed=2, in_count=8))
    assert len(analyze.blocks_of(64)) == 16


def test_extended_audit_refuses_missing_or_off_design_runs(tmp_path):
    write_runs(tmp_path / "short", "base", range(32), 32)
    write_runs(tmp_path / "short", "more", range(32, 63), 64)
    with pytest.raises(ValueError, match="missing"):
        analyze.load_extended(tmp_path / "short", "base", "more")
    write_runs(tmp_path / "off", "base", range(32), 32)
    write_runs(tmp_path / "off", "more", range(32, 64), 64, flip=(0, 40))
    with pytest.raises(ValueError, match="differs"):
        analyze.load_extended(tmp_path / "off", "base", "more")
