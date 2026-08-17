"""Tests for the autonomous per-curve noise-search state machine.

All tests here use a fake ``run_and_score`` callable -- no real training.
"""

from __future__ import annotations

from experiments.cia.scripts.auc_target_search import (
    StageResult,
    find_low_noise_anchor,
    step_up_to_target,
)


def _fake_runner(auc_by_ratio: dict[float, float], accuracy: float = 0.8):
    """Build a run_and_score fake that looks up AUC by exact ratio."""

    def run_and_score(ratio: float) -> StageResult:
        return StageResult(
            noise_ratio=ratio, seed=42, accuracy=accuracy, auc=auc_by_ratio[ratio]
        )

    return run_and_score


def test_find_low_noise_anchor_returns_first_stage_within_tolerance() -> None:
    # Starting ratio already matches vanilla -- no halving needed.
    run_and_score = _fake_runner({0.01: 0.73}, accuracy=0.90)
    anchor, stages = find_low_noise_anchor(
        starting_ratio=0.01,
        vanilla_accuracy=0.90,
        vanilla_auc=0.73,
        anchor_tolerance=0.03,
        max_halvings=4,
        run_and_score=run_and_score,
    )
    assert anchor == 0.01
    assert len(stages) == 1


def test_find_low_noise_anchor_halves_until_convergence() -> None:
    # vanilla_auc=0.74, tolerance=0.03 -> converges once auc lands in [0.71, 0.77].
    # 0.08/0.04/0.02 all miss (diffs 0.14/0.08/0.045); 0.01 lands (diff 0.005).
    run_and_score = _fake_runner(
        {0.08: 0.60, 0.04: 0.66, 0.02: 0.695, 0.01: 0.735}, accuracy=0.90
    )
    anchor, stages = find_low_noise_anchor(
        starting_ratio=0.08,
        vanilla_accuracy=0.90,
        vanilla_auc=0.74,
        anchor_tolerance=0.03,
        max_halvings=4,
        run_and_score=run_and_score,
    )
    assert anchor == 0.01
    assert len(stages) == 4


def test_find_low_noise_anchor_gives_up_after_max_halvings() -> None:
    run_and_score = _fake_runner(
        {0.16: 0.50, 0.08: 0.50, 0.04: 0.50, 0.02: 0.50, 0.01: 0.50}, accuracy=0.90
    )
    anchor, stages = find_low_noise_anchor(
        starting_ratio=0.16,
        vanilla_accuracy=0.90,
        vanilla_auc=0.95,
        anchor_tolerance=0.03,
        max_halvings=4,
        run_and_score=run_and_score,
    )
    assert anchor is None
    assert len(stages) == 5  # starting stage + 4 halvings


def test_step_up_to_target_lands_in_band() -> None:
    run_and_score = _fake_runner({0.01: 0.72, 0.02: 0.60, 0.04: 0.49}, accuracy=0.80)
    status, landing_ratio, stages = step_up_to_target(
        anchor_ratio=0.01,
        target_band=(0.45, 0.55),
        step_multiplier=2.0,
        stage_cap=12,
        random_baseline_accuracy=0.10,
        collapse_margin=0.02,
        run_and_score=run_and_score,
    )
    assert status == "landed"
    assert landing_ratio == 0.04
    assert len(stages) == 3


def test_step_up_to_target_detects_collapse_before_landing() -> None:
    # Explicit stage sequence: last stage collapses (accuracy near baseline)
    # before AUC ever reaches the target band.
    stages_by_ratio = {
        0.01: StageResult(noise_ratio=0.01, seed=42, accuracy=0.80, auc=0.72),
        0.02: StageResult(noise_ratio=0.02, seed=42, accuracy=0.60, auc=0.65),
        0.04: StageResult(noise_ratio=0.04, seed=42, accuracy=0.11, auc=0.58),
    }

    def run_and_score(ratio: float) -> StageResult:
        return stages_by_ratio[ratio]

    status, landing_ratio, stages = step_up_to_target(
        anchor_ratio=0.01,
        target_band=(0.45, 0.55),
        step_multiplier=2.0,
        stage_cap=12,
        random_baseline_accuracy=0.10,
        collapse_margin=0.02,
        run_and_score=run_and_score,
    )
    assert status == "collapsed-before-target"
    assert landing_ratio is None
    assert len(stages) == 3


def test_step_up_to_target_reaches_stage_cap() -> None:
    def run_and_score(ratio: float) -> StageResult:
        return StageResult(noise_ratio=ratio, seed=42, accuracy=0.80, auc=0.70)

    status, landing_ratio, stages = step_up_to_target(
        anchor_ratio=0.01,
        target_band=(0.45, 0.55),
        step_multiplier=2.0,
        stage_cap=3,
        random_baseline_accuracy=0.10,
        collapse_margin=0.02,
        run_and_score=run_and_score,
    )
    assert status == "stage-cap-reached"
    assert landing_ratio is None
    assert len(stages) == 3


def test_step_up_to_target_collapse_takes_priority_over_landing() -> None:
    # A single stage whose AUC already sits inside the target band but whose
    # accuracy has also collapsed to the random baseline must be reported as
    # a collapse, not a landing -- the collapse check must run before (and
    # win over) the landing check within one stage.
    def run_and_score(ratio: float) -> StageResult:
        return StageResult(noise_ratio=ratio, seed=42, accuracy=0.10, auc=0.50)

    status, landing_ratio, stages = step_up_to_target(
        anchor_ratio=0.01,
        target_band=(0.45, 0.55),
        step_multiplier=2.0,
        stage_cap=12,
        random_baseline_accuracy=0.10,
        collapse_margin=0.02,
        run_and_score=run_and_score,
    )
    assert status == "collapsed-before-target"
    assert landing_ratio is None
    assert len(stages) == 1
