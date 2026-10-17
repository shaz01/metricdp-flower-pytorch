"""CIA roadmap v2 attack library: calibrated loss attacker, metrics, two-way bootstrap, design matrix.

Includes the G-M0 Gaussian calibration test: on synthetic trajectories whose IN and OUT
distributions are Gaussian with Mahalanobis separation ``mu`` (a mu-GDP pair), the pipeline
must recover ``AUC = Phi(mu / sqrt 2)`` and ``TPR(alpha) = Phi(mu - Phi^-1(1 - alpha))`` within
its bootstrap interval.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from experiments.cia.audit_attacks import (
    balanced_references,
    Interval,
    calibrated_scores,
    difference_bootstrap,
    pair_jackknife,
    pair_jackknife_difference,
    roc_auc,
    threshold_scores,
    tpr_at_fpr,
    two_way_bootstrap,
)
from experiments.cia.inclusion import balanced_design

CANDIDATES = 16
CHECKPOINTS = 11


def synthetic(mu, replicates, seed, checkpoints=CHECKPOINTS, rho=0.8):
    """Loss trajectories: AR(1) noise, per-candidate difficulty, IN shift with Mahalanobis norm ``mu``."""
    rng = np.random.default_rng(seed)
    lags = np.abs(np.subtract.outer(np.arange(checkpoints), np.arange(checkpoints)))
    cov = rho ** lags
    chol = np.linalg.cholesky(cov)
    direction = np.linspace(1.0, 2.0, checkpoints)  # leakage grows over training
    shift = direction * mu / np.sqrt(direction @ np.linalg.solve(cov, direction))
    membership = balanced_design(CANDIDATES, replicates, seed=seed)
    difficulty = rng.normal(0.0, 3.0, size=(CANDIDATES, 1)) * direction
    noise = rng.standard_normal((CANDIDATES, replicates, checkpoints)) @ chol.T
    losses = difficulty[:, None, :] + noise - membership[..., None] * shift  # IN lowers the loss
    return losses, membership


def test_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    scores, labels = rng.normal(size=300).round(1), rng.random(300) < 0.4
    assert roc_auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))


def test_tpr_at_fpr_by_hand():
    scores = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
    labels = np.array([1, 0, 1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    # At most 1 of 6 negatives: scores >= 6 (negative 8 only) -> TPR 3/4; at most 0: scores >= 9 -> 1/4.
    assert tpr_at_fpr(scores, labels, 1 / 6) == pytest.approx(0.75)
    assert tpr_at_fpr(scores, labels, 0.1) == pytest.approx(0.25)


def test_tpr_at_fpr_does_not_split_ties():
    scores = np.array([1.0, 1.0, 0.0])
    labels = np.array([True, False, False])
    assert tpr_at_fpr(scores, labels, 0.4) == 0.0


def test_threshold_scores_rank_lower_final_loss_as_in():
    losses = np.array([[[3.0, 1.0], [3.0, 2.0]]])
    assert threshold_scores(losses).tolist() == [[-1.0, -2.0]]


def test_calibrated_score_ignores_the_trials_own_membership():
    losses, membership = synthetic(1.0, 8, seed=1)
    flipped = membership.copy()
    flipped[3, 5] = ~flipped[3, 5]
    a, b = calibrated_scores(losses, membership), calibrated_scores(losses, flipped)
    assert a[3, 5] == pytest.approx(b[3, 5])


def test_calibrated_score_removes_candidate_difficulty():
    losses, membership = synthetic(1.0, 8, seed=2)
    shifted = losses.copy()
    shifted[0] += 100.0
    np.testing.assert_allclose(calibrated_scores(losses, membership)[0],
                               calibrated_scores(shifted, membership)[0], atol=1e-8)


@pytest.mark.parametrize("fixed_count", [True, False])
def test_references_exclude_the_trial_and_balance_the_worlds(fixed_count):
    membership = balanced_design(CANDIDATES, 8, seed=3, fixed_count=fixed_count)
    for trial in range(8):
        references = balanced_references(membership, trial)
        assert not references[:, trial].any()
        assert ((membership & references).sum(axis=1) == (~membership & references).sum(axis=1)).all()
        assert ((membership & references).sum(axis=1) == 3).all()


@pytest.mark.parametrize("fixed_count", [True, False])
def test_no_leakage_is_unbiased_on_average(fixed_count):
    """Leaving out only the trial gives a mean null AUC of about 0.42 at R = 8 (unequal reference counts)."""
    aucs = []
    for seed in range(200):
        losses, _ = synthetic(0.0, 8, seed=1000 + seed)
        membership = balanced_design(CANDIDATES, 8, seed=seed, fixed_count=fixed_count)
        aucs.append(roc_auc(calibrated_scores(losses, membership).ravel(), membership.ravel()))
    assert np.mean(aucs) == pytest.approx(0.5, abs=0.015)


def test_no_leakage_gives_an_interval_around_one_half():
    hits = 0
    for seed in range(20):
        losses, membership = synthetic(0.0, 8, seed=100 + seed)
        result = two_way_bootstrap(calibrated_scores(losses, membership), membership, roc_auc,
                                   resamples=500, seed=seed)
        hits += result.low <= 0.5 <= result.high
    assert hits >= 17  # 95% intervals: 19 of 20 expected


@pytest.mark.parametrize("mu", [1.0, 2.0])
def test_calibration_recovers_the_gaussian_tradeoff(mu):
    """G-M0 calibration test (ROADMAP.md Section 2, item 7).

    1,000 replicates make the attacker's reference estimates nearly exact, so the pipeline
    (leave-one-replicate-out likelihood ratio, metrics, two-way bootstrap) is tested rather
    than the attacker's finite-sample efficiency, which the power table measures at R = 8-16.
    """
    losses, membership = synthetic(mu, 1000, seed=7)
    scores = calibrated_scores(losses, membership)
    auc = two_way_bootstrap(scores, membership, roc_auc, resamples=400, seed=0)
    assert auc.low <= norm.cdf(mu / np.sqrt(2)) <= auc.high
    for alpha in (0.05, 0.10):
        tpr = two_way_bootstrap(scores, membership, lambda s, m: tpr_at_fpr(s, m, alpha), resamples=400, seed=0)
        assert tpr.low <= norm.cdf(mu - norm.ppf(1 - alpha)) <= tpr.high


def test_pooled_direction_beats_a_per_candidate_direction_at_eight_replicates():
    projected, full = [], []
    for seed in range(30):
        losses, membership = synthetic(1.0, 8, seed=300 + seed)
        projected.append(roc_auc(calibrated_scores(losses, membership).ravel(), membership.ravel()))
        full.append(roc_auc(calibrated_scores(losses, membership, project=False).ravel(), membership.ravel()))
    assert np.mean(projected) > np.mean(full) + 0.04


def test_final_round_only_attacker():
    losses, membership = synthetic(2.0, 200, seed=3, checkpoints=1)
    assert roc_auc(calibrated_scores(losses, membership).ravel(), membership.ravel()) > 0.8


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible():
    losses, membership = synthetic(1.0, 8, seed=4)
    scores = calibrated_scores(losses, membership)
    first = two_way_bootstrap(scores, membership, roc_auc, resamples=300, seed=5)
    second = two_way_bootstrap(scores, membership, roc_auc, resamples=300, seed=5)
    assert first == second
    assert first.low <= first.estimate <= first.high
    assert first.estimate == pytest.approx(roc_auc(scores.ravel(), membership.ravel()))


@pytest.mark.parametrize("replicates", [8, 12, 16])
def test_balanced_design_fixes_row_and_column_counts(replicates):
    design = balanced_design(CANDIDATES, replicates, seed=0)
    assert design.shape == (CANDIDATES, replicates)
    assert (design.sum(axis=1) == replicates // 2).all()
    assert (design.sum(axis=0) == CANDIDATES // 2).all()
    np.testing.assert_array_equal(design, balanced_design(CANDIDATES, replicates, seed=0))
    assert not np.array_equal(design, balanced_design(CANDIDATES, replicates, seed=1))


def test_variable_count_design_balances_rows_only():
    design = balanced_design(CANDIDATES, 8, seed=0, fixed_count=False)
    assert (design.sum(axis=1) == 4).all()
    assert len(set(design.sum(axis=0).tolist())) > 1


def test_balanced_design_needs_an_even_replicate_count():
    with pytest.raises(ValueError):
        balanced_design(CANDIDATES, 7, seed=0)


def test_difference_of_identical_configurations_is_exactly_zero():
    losses, membership = synthetic(1.0, 8, seed=6)
    scores = calibrated_scores(losses, membership)
    result = difference_bootstrap(scores, membership, scores, membership, roc_auc, resamples=200, seed=0)
    assert result == Interval(0.0, 0.0, 0.0)


def test_difference_detects_a_large_leakage_gap():
    strong, member = synthetic(3.0, 16, seed=8)
    weak, _ = synthetic(0.0, 16, seed=8)
    result = difference_bootstrap(calibrated_scores(strong, member), member, calibrated_scores(weak, member), member,
                                  roc_auc, resamples=300, seed=0)
    assert result.low > 0
    assert result.estimate == pytest.approx(roc_auc(calibrated_scores(strong, member), member)
                                            - roc_auc(calibrated_scores(weak, member), member))


def test_difference_needs_one_design():
    _, member = synthetic(0.0, 8, seed=1)
    with pytest.raises(ValueError):
        difference_bootstrap(np.zeros((16, 8)), member, np.zeros((16, 6)), member[:, :6], roc_auc)


def _jackknife_by_hand(values, estimate, level=0.95):
    from scipy.stats import t as student

    n = len(values)
    se = np.sqrt((n - 1) / n * np.sum((values - values.mean()) ** 2))
    q = student.ppf(0.5 + level / 2, n - 1)
    return estimate - q * se, estimate + q * se


def test_pair_jackknife_matches_a_hand_computation():
    losses, membership = synthetic(1.0, 8, seed=12)
    scores = calibrated_scores(losses, membership)
    keep = [np.setdiff1d(np.arange(8), [2 * p, 2 * p + 1]) for p in range(4)]
    values = np.array([roc_auc(scores[:, k], membership[:, k]) for k in keep])
    low, high = _jackknife_by_hand(values, roc_auc(scores, membership))
    result = pair_jackknife(scores, membership, roc_auc)
    assert (result.low, result.high) == pytest.approx((low, high))
    assert result.estimate == pytest.approx(roc_auc(scores, membership))


def test_pair_jackknife_difference_of_identical_configurations_is_zero():
    losses, membership = synthetic(1.0, 8, seed=13)
    scores = calibrated_scores(losses, membership)
    assert pair_jackknife_difference(scores, membership, scores, membership, roc_auc) == Interval(0.0, 0.0, 0.0)


def test_pair_jackknife_needs_complementary_pairs():
    membership = balanced_design(CANDIDATES, 8, seed=0, fixed_count=False)
    with pytest.raises(ValueError, match="complementary"):
        pair_jackknife(np.zeros((CANDIDATES, 8)), membership, roc_auc)
