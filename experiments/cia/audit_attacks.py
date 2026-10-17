"""Attackers and metrics of the CIA measurement standard v2 (research/cia_roadmap_v2/ROADMAP.md Section 2).

Trials are ``(candidate, replicate)`` pairs. ``losses`` has shape ``(candidates, replicates,
checkpoints)`` (a candidate's shadow-sample loss at each scored round) and ``membership`` shape
``(candidates, replicates)`` (True where the candidate took part). Every score is higher for IN.
References are leave-one-replicate-out: trial ``(c, j)`` is scored only from replicates other
than ``j``, so its own world never informs its score.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import rankdata
from sklearn.covariance import ledoit_wolf_shrinkage


def balanced_references(membership: np.ndarray, trial: int) -> np.ndarray:
    """Per candidate, the replicates that may serve as references for scoring replicate ``trial``.

    Leaving out only ``trial`` leaves ``R/2 - 1`` references in the trial's own world and ``R/2``
    in the other, and that imbalance alone biases the likelihood ratio: under no leakage the
    pooled AUC is about 0.42 at R = 8, because the midpoint between the two reference means
    leans towards the noisier one. Equal counts remove the bias, so each candidate also drops
    one replicate of the larger world, the nearest to ``trial`` in cyclic order (in the
    fixed-count design that is the trial's complementary partner whenever it is adjacent).
    """
    member = np.asarray(membership, dtype=bool)
    candidates, replicates = member.shape
    references = np.ones_like(member)
    references[:, trial] = False
    order = sorted(range(replicates), key=lambda r: (min((r - trial) % replicates, (trial - r) % replicates), r))
    for c in range(candidates):
        n_in = int((member[c] & references[c]).sum())
        n_out = int((~member[c] & references[c]).sum())
        while n_in != n_out:
            larger = n_in > n_out
            drop = next(r for r in order if references[c, r] and member[c, r] == larger)
            references[c, drop] = False
            n_in, n_out = (n_in - 1, n_out) if larger else (n_in, n_out - 1)
    return references


def calibrated_scores(losses: np.ndarray, membership: np.ndarray, *, project: bool = True) -> np.ndarray:
    """Gaussian log-likelihood ratio per trial, LiRA style with per-candidate difficulty calibration.

    For trial ``(c, j)``, from the references of ``balanced_references`` only:

    1. IN and OUT mean trajectories of every candidate, and one covariance ``S`` pooled over
       candidates and both worlds (within-group residuals, Ledoit-Wolf shrinkage towards a
       scaled identity);
    2. a direction ``w = S^-1 dbar``, with ``dbar`` the IN-OUT mean difference averaged over
       candidates; trajectories are projected on ``w``;
    3. a per-candidate one-dimensional Gaussian likelihood ratio on the projection,
       ``(m_IN - m_OUT) (y - (m_IN + m_OUT) / 2) / (w^T S w)``.

    Pooling the direction matters at the design sizes: with three references per world, a
    per-candidate 11-dimensional ratio (``project=False``) is dominated by the noise of its own
    mean estimates (synthetic mu = 1: AUC 0.59 at R = 8 against 0.68 projected and 0.76 optimal).
    The noise-level attacker uses this function with run-level features broadcast to all candidates.
    """
    x = np.asarray(losses, dtype=np.float64)
    if x.ndim == 2:
        x = x[..., None]
    member = np.asarray(membership, dtype=bool)
    candidates, replicates, checkpoints = x.shape
    if member.shape != (candidates, replicates):
        raise ValueError("membership must have shape (candidates, replicates)")
    scores = np.empty((candidates, replicates))
    for j in range(replicates):
        references = balanced_references(member, j)
        weight_in = (member & references).astype(np.float64)
        weight_out = (~member & references).astype(np.float64)
        n_in, n_out = weight_in.sum(axis=1), weight_out.sum(axis=1)
        if (n_in < 1).any():
            raise ValueError("every candidate needs IN and OUT references besides each trial")
        mean_in = np.einsum("cr,crk->ck", weight_in, x) / n_in[:, None]
        mean_out = np.einsum("cr,crk->ck", weight_out, x) / n_out[:, None]
        residuals = (x - np.where(member[..., None], mean_in[:, None], mean_out[:, None]))[references]
        dof = len(residuals) - 2 * candidates
        sample = residuals.T @ residuals / dof
        shrinkage = ledoit_wolf_shrinkage(residuals, assume_centered=True) if checkpoints > 1 else 0.0
        covariance = (1 - shrinkage) * sample + shrinkage * np.trace(sample) / checkpoints * np.eye(checkpoints)
        if project:
            w = np.linalg.solve(covariance, (mean_in - mean_out).mean(axis=0))
            projected_in, projected_out = mean_in @ w, mean_out @ w
            scores[:, j] = ((projected_in - projected_out) * (x[:, j] @ w - (projected_in + projected_out) / 2)
                            / (w @ covariance @ w))
        else:
            direction = np.linalg.solve(covariance, (mean_in - mean_out).T).T
            scores[:, j] = np.einsum("ck,ck->c", direction, x[:, j] - (mean_in + mean_out) / 2)
    return scores


def threshold_scores(losses: np.ndarray) -> np.ndarray:
    """Uncalibrated baseline: the final checkpoint's loss, lower loss meaning IN."""
    return -np.asarray(losses, dtype=np.float64)[..., -1]


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve (Mann-Whitney, ties counted one half)."""
    scores, labels = np.ravel(scores), np.ravel(labels).astype(bool)
    positives, negatives = labels.sum(), (~labels).sum()
    if not positives or not negatives:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def tpr_at_fpr(scores: np.ndarray, labels: np.ndarray, fpr: float) -> float:
    """Largest TPR of a threshold rule ``score >= t`` whose FPR is at most ``fpr`` (ties never split)."""
    scores, labels = np.ravel(scores), np.ravel(labels).astype(bool)
    if not labels.any() or labels.all():
        return float("nan")
    negatives = np.sort(scores[~labels])
    positives = np.sort(scores[labels])
    thresholds = np.unique(scores)  # ascending: the lowest admissible threshold maximizes TPR
    false_rate = 1.0 - np.searchsorted(negatives, thresholds, side="left") / len(negatives)
    admissible = thresholds[false_rate <= fpr + 1e-12]
    if not len(admissible):
        return 0.0
    return float(1.0 - np.searchsorted(positives, admissible[0], side="left") / len(positives))


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


def _resampled_indices(candidates: int, replicates: int, resamples: int, seed: int):
    """Two-way bootstrap index sets: candidates and replicates drawn independently, with replacement."""
    rng = np.random.default_rng(seed)
    for _ in range(resamples):
        yield np.ix_(rng.integers(candidates, size=candidates), rng.integers(replicates, size=replicates))


def _interval(estimate: float, values: np.ndarray, level: float) -> Interval:
    tail = 100 * (1 - level) / 2
    low, high = np.nanpercentile(values, [tail, 100 - tail])
    return Interval(float(estimate), float(low), float(high))


def two_way_bootstrap(
    scores: np.ndarray,
    membership: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    resamples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """Percentile interval from resampling candidates and replicates independently, with replacement.

    Each resample keeps the trials of the drawn candidates in the drawn replicates (with
    multiplicity) and recomputes ``metric`` on them.
    """
    scores, member = np.asarray(scores), np.asarray(membership, dtype=bool)
    values = np.array([metric(scores[index].ravel(), member[index].ravel())
                       for index in _resampled_indices(*scores.shape, resamples, seed)])
    return _interval(metric(scores.ravel(), member.ravel()), values, level)


def difference_bootstrap(
    scores_a: np.ndarray,
    membership_a: np.ndarray,
    scores_b: np.ndarray,
    membership_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    resamples: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> Interval:
    """Interval for ``metric(a) - metric(b)`` over two configurations of one inclusion design.

    Both configurations are resampled with the same candidate and replicate draws: their runs
    share the design (and, by default, the replicate seeds), so replicate ``j`` of ``a`` and of
    ``b`` form a pair.
    """
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    member_a, member_b = np.asarray(membership_a, dtype=bool), np.asarray(membership_b, dtype=bool)
    if a.shape != b.shape or member_a.shape != a.shape or member_b.shape != b.shape:
        raise ValueError("both configurations need scores and membership of one design shape")
    values = np.array([metric(a[index].ravel(), member_a[index].ravel())
                       - metric(b[index].ravel(), member_b[index].ravel())
                       for index in _resampled_indices(*a.shape, resamples, seed)])
    estimate = metric(a.ravel(), member_a.ravel()) - metric(b.ravel(), member_b.ravel())
    return _interval(estimate, values, level)


def _complementary_pairs(membership: np.ndarray) -> list[np.ndarray]:
    member = np.asarray(membership, dtype=bool)
    replicates = member.shape[1]
    if replicates % 2 or any((member[:, 2 * p] == member[:, 2 * p + 1]).any() for p in range(replicates // 2)):
        raise ValueError("the pair jackknife needs the fixed-count design's complementary replicate pairs")
    return [np.array([2 * p, 2 * p + 1]) for p in range(replicates // 2)]


def _jackknife_interval(estimate: float, values: np.ndarray, level: float) -> Interval:
    from scipy.stats import t as student

    n = len(values)
    se = float(np.sqrt((n - 1) / n * np.sum((values - values.mean()) ** 2)))
    q = float(student.ppf(0.5 + level / 2, n - 1))
    return Interval(float(estimate), float(estimate - q * se), float(estimate + q * se))


def pair_jackknife(
    scores: np.ndarray,
    membership: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    level: float = 0.95,
) -> Interval:
    """Gate interval for one configuration: leave-one-pair-out jackknife with t quantiles (P - 1 df).

    The unit is a complementary replicate pair of the fixed-count design. The candidates are a fixed panel (the
    interval does not generalise beyond them; ``two_way_bootstrap`` does, conservatively). Calibrated by simulation
    at 32 candidates and R = 32 (``results/cia_v2/measurement/power_options.py``); liberal at R = 16.
    """
    scores, member = np.asarray(scores), np.asarray(membership, dtype=bool)
    pairs = _complementary_pairs(member)
    keep = [np.setdiff1d(np.arange(scores.shape[1]), pair) for pair in pairs]
    values = np.array([metric(scores[:, k].ravel(), member[:, k].ravel()) for k in keep])
    return _jackknife_interval(metric(scores.ravel(), member.ravel()), values, level)


def pair_jackknife_difference(
    scores_a: np.ndarray,
    membership_a: np.ndarray,
    scores_b: np.ndarray,
    membership_b: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    level: float = 0.95,
) -> Interval:
    """Gate interval for ``metric(a) - metric(b)``, two configurations of one design (same pairs left out)."""
    a, b = np.asarray(scores_a), np.asarray(scores_b)
    member_a, member_b = np.asarray(membership_a, dtype=bool), np.asarray(membership_b, dtype=bool)
    if a.shape != b.shape or member_a.shape != a.shape or member_b.shape != b.shape:
        raise ValueError("both configurations need scores and membership of one design shape")
    pairs = _complementary_pairs(member_a)
    keep = [np.setdiff1d(np.arange(a.shape[1]), pair) for pair in pairs]
    values = np.array([metric(a[:, k].ravel(), member_a[:, k].ravel()) - metric(b[:, k].ravel(), member_b[:, k].ravel())
                       for k in keep])
    estimate = metric(a.ravel(), member_a.ravel()) - metric(b.ravel(), member_b.ravel())
    return _jackknife_interval(estimate, values, level)
