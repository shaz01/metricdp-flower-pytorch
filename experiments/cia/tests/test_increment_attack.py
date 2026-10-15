"""Synthetic tests for the defense-aware CIA scorers (WP1)."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from experiments.cia.increment_attack import (
    blind_sigma,
    increment_llr,
    increment_scores,
    loss_scores,
    noise_level_scores,
    pooled,
    reference_moments,
    roc_auc,
    stratified_cluster_bootstrap,
    tpr_at_fpr,
)


def test_blind_sigma_and_noise_level_scores_by_hand() -> None:
    np.testing.assert_allclose(blind_sigma([6.0, 2.0], 4), [3.0, 1.0])
    scores = noise_level_scores([1.0, 2.0], np.array([[2.0, 2.0], [2.0, 2.0]]))
    np.testing.assert_allclose(scores, [math.log(2.0), 0.0])
    with pytest.raises(ValueError):
        blind_sigma([1.0], 0)
    with pytest.raises(ValueError):
        noise_level_scores([1.0], np.array([[1.0, 1.0]]))


def test_loss_scores_lower_loss_is_more_in_like() -> None:
    np.testing.assert_allclose(loss_scores([1.0, 3.0], np.array([[2.0, 2.0], [4.0, 2.0]])), [2.0, -1.0])


def test_reference_moments_by_hand() -> None:
    mean_in, mean_out, tau2 = reference_moments(np.array([[1.0, 2.0], [3.0, 4.0]]),
                                                np.array([[0.0, 0.0], [2.0, 2.0]]))
    np.testing.assert_allclose(mean_in, [2.0, 3.0])
    np.testing.assert_allclose(mean_out, [1.0, 1.0])
    assert tau2 == pytest.approx(8.0 / 4.0)  # squares 4 + 4 over (1 + 1) * 2 rounds
    assert reference_moments(np.ones((1, 3)), np.zeros((1, 3)))[2] == 0.0


def test_increment_llr_by_hand_and_cumulative_score() -> None:
    reference_in = np.array([[1.0, 2.0], [3.0, 4.0]])   # mean (2, 3)
    reference_out = np.array([[0.0, 0.0], [2.0, 2.0]])  # mean (1, 1); tau^2 = 2
    llr = increment_llr([2.0, 1.0], [1.0, 1.0], [0.0, 1.0], reference_in, reference_out)
    # v = (0 + 2, 1 + 2); ((d - 1)^2 - (d - mu_in)^2) / (2 v)
    np.testing.assert_allclose(llr, [1.0 / 4.0, -4.0 / 6.0])
    np.testing.assert_allclose(
        increment_scores([2.0, 1.0], [1.0, 1.0], [0.0, 1.0], reference_in, reference_out),
        [0.25, 0.25 - 4.0 / 6.0],
    )


@pytest.mark.parametrize("references, low, high", [(2000, 0.91, 0.97), (10, 0.80, 0.97)])
def test_increment_attack_reaches_the_gaussian_optimum(references, low, high) -> None:
    """IN increments are 0.5 lower per round than OUT, unit noise, 20 rounds.

    With known means the summed log-likelihood ratio is optimal: AUC = Phi(0.5 sqrt(20) / sqrt(2))
    = 0.943. Many references recover it; ten references per world lose some power.
    """
    rng = np.random.default_rng(0)
    rounds = 20
    shift = np.full(rounds, -0.5)
    references_in = shift + rng.normal(size=(references, rounds))
    references_out = rng.normal(size=(references, rounds))
    ones, zeros = np.ones(rounds), np.zeros(rounds)
    in_scores = [increment_scores(shift + rng.normal(size=rounds), ones, zeros,
                                  references_in, references_out)[-1] for _ in range(400)]
    out_scores = [increment_scores(rng.normal(size=rounds), ones, zeros,
                                   references_in, references_out)[-1] for _ in range(400)]
    assert low < roc_auc(in_scores, out_scores) < high


def test_roc_auc_matches_brute_force_with_ties() -> None:
    assert roc_auc([3.0, 2.0], [1.0, 2.0]) == pytest.approx(3.5 / 4.0)
    rng = np.random.default_rng(1)
    positive, negative = rng.integers(0, 5, 30).astype(float), rng.integers(0, 5, 40).astype(float)
    brute = np.mean([1.0 if p > n else 0.5 if p == n else 0.0
                     for p, n in itertools.product(positive, negative)])
    assert roc_auc(positive, negative) == pytest.approx(brute)


def test_tpr_at_fpr_by_hand() -> None:
    positive, negative = [5.0, 4.0, 3.0], [4.0, 2.0, 1.0, 0.0]
    assert tpr_at_fpr(positive, negative, 0.25) == pytest.approx(1.0)
    assert tpr_at_fpr(positive, negative, 0.2) == pytest.approx(1.0 / 3.0)
    assert tpr_at_fpr(positive, negative, 0.0) == pytest.approx(1.0 / 3.0)
    assert tpr_at_fpr([1.0], [1.0], 0.5) == 0.0  # a tied negative cannot be split off


def test_stratified_cluster_bootstrap_is_reproducible_and_keeps_strata() -> None:
    rng = np.random.default_rng(2)
    strata = [[(rng.normal(1, 1, 5), rng.normal(0, 1, 5)) for _ in range(20)] for _ in range(2)]
    statistic = pooled(roc_auc)
    first = stratified_cluster_bootstrap(strata, statistic, resamples=200)
    second = stratified_cluster_bootstrap(strata, statistic, resamples=200)
    assert first == second
    point, (low, high) = first
    assert low <= point <= high and point > 0.7
    sizes = []
    stratified_cluster_bootstrap(strata, lambda selected: sizes.append([len(s) for s in selected]) or 0.0,
                                 resamples=3)
    assert all(size == [20, 20] for size in sizes)
