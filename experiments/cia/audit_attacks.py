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
    candidates, replicates = scores.shape
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    for b in range(resamples):
        rows, columns = rng.integers(candidates, size=candidates), rng.integers(replicates, size=replicates)
        index = np.ix_(rows, columns)
        values[b] = metric(scores[index].ravel(), member[index].ravel())
    tail = 100 * (1 - level) / 2
    low, high = np.nanpercentile(values, [tail, 100 - tail])
    return Interval(float(metric(scores.ravel(), member.ravel())), float(low), float(high))
