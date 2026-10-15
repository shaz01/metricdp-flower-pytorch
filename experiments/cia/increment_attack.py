"""Defense-aware CIA scorers: noise-level and per-round increment likelihood-ratio attacks.

Pure numpy, no training. Conventions (fixed before any run, see
``results/cia_frontier/side_channel_audit/PROTOCOL.md``):

* every score is oriented so that **higher means more IN-like**;
* per-round series are arrays of shape ``(rounds,)``; reference sets are arrays of
  shape ``(references, rounds)`` taken from trajectories the attacker can simulate
  (e.g. the same federation with other training seeds), never from the observed one;
* the observed trajectory sees the released global models ``theta_0 .. theta_R``,
  the target's shadow sample and the public protocol (mechanism, clipping norm).

Noise-level attack: the released update is ``Delta_t = s_t + xi_t`` with
``xi_t ~ N(0, sigma_t^2 I_D)``, so ``||Delta_t|| / sqrt(D)`` estimates ``sigma_t``
(blind: the signal ``s_t`` is not removed). Lower noise than the reference federations
without the target means IN (metric privacy: the target widens the maximum pairwise
distance ``d_t`` and ``sigma_t = z C / (n d_t)`` falls).

Increment attack: the shadow-loss increment ``delta_t = L(theta_t) - L(theta_{t-1})``
is modelled per round as Gaussian with world mean taken from the references and variance
``sigma_t^2 ||g_t||^2 + tau^2``: the first-order loss change caused by the release noise
(``g_t`` = shadow-loss gradient at ``theta_{t-1}``) plus the pooled residual variance of
the reference increments. The round's log-likelihood ratio is IN versus OUT; the score
at round ``r`` is the cumulative sum over rounds ``1..r``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

_MIN_VARIANCE = 1e-12


def blind_sigma(update_norms: Sequence[float], parameter_count: int) -> np.ndarray:
    """Blind per-round noise std estimate ``||theta_t - theta_{t-1}|| / sqrt(D)``."""
    if parameter_count < 1:
        raise ValueError("parameter_count must be positive")
    norms = np.asarray(update_norms, dtype=float)
    if np.any(norms < 0) or not np.all(np.isfinite(norms)):
        raise ValueError("update norms must be finite and non-negative")
    return norms / np.sqrt(parameter_count)


def _references(values: np.ndarray, rounds: int, name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != rounds:
        raise ValueError(f"{name} must have shape (references, {rounds})")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    return values


def noise_level_scores(observed_sigma: Sequence[float], reference_out_sigma: np.ndarray) -> np.ndarray:
    """Per-round score ``-(log sigma_obs - mean_ref log sigma_ref)``; higher = lower noise = IN."""
    observed = np.asarray(observed_sigma, dtype=float)
    references = _references(reference_out_sigma, len(observed), "reference_out_sigma")
    if np.any(observed <= 0) or np.any(references <= 0):
        raise ValueError("sigma estimates must be positive")
    return -(np.log(observed) - np.log(references).mean(axis=0))


def loss_scores(observed_loss: Sequence[float], reference_loss: np.ndarray) -> np.ndarray:
    """Current (loss-only) statistic per round: ``-(L_obs - mean_ref L)``; lower loss = IN.

    ``reference_loss`` pools reference trajectories of both worlds, so the offset does
    not favour either hypothesis; it only puts targets and rounds on one scale.
    """
    observed = np.asarray(observed_loss, dtype=float)
    references = _references(reference_loss, len(observed), "reference_loss")
    return -(observed - references.mean(axis=0))


def reference_moments(reference_in: np.ndarray, reference_out: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Per-round world means and the pooled residual variance ``tau^2`` of the references.

    ``tau^2`` pools squared deviations from the per-round world means over rounds and both
    worlds, with ``references - 1`` degrees of freedom per world and round. With a single
    reference in a world that world contributes no degrees of freedom.
    """
    reference_in = np.asarray(reference_in, dtype=float)
    if reference_in.ndim != 2:
        raise ValueError("reference_in must have shape (references, rounds)")
    rounds = reference_in.shape[1]
    reference_in = _references(reference_in, rounds, "reference_in")
    reference_out = _references(reference_out, rounds, "reference_out")
    mean_in, mean_out = reference_in.mean(axis=0), reference_out.mean(axis=0)
    squares = ((reference_in - mean_in) ** 2).sum() + ((reference_out - mean_out) ** 2).sum()
    dof = (reference_in.shape[0] - 1 + reference_out.shape[0] - 1) * rounds
    tau2 = float(squares / dof) if dof > 0 else 0.0
    return mean_in, mean_out, tau2


def increment_llr(
    increments: Sequence[float],
    gradient_norms: Sequence[float],
    sigma: Sequence[float],
    reference_in: np.ndarray,
    reference_out: np.ndarray,
) -> np.ndarray:
    """Per-round log-likelihood ratio log N(delta; mu_IN, v) - log N(delta; mu_OUT, v).

    ``v_t = sigma_t^2 ||g_t||^2 + tau^2``; with a shared variance the ratio reduces to
    ``((delta - mu_OUT)^2 - (delta - mu_IN)^2) / (2 v)``.
    """
    delta = np.asarray(increments, dtype=float)
    gradient = np.asarray(gradient_norms, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if not (delta.shape == gradient.shape == sigma.shape) or delta.ndim != 1:
        raise ValueError("increments, gradient_norms and sigma must be equal-length vectors")
    if np.any(gradient < 0) or np.any(sigma < 0):
        raise ValueError("gradient norms and sigma must be non-negative")
    mean_in, mean_out, tau2 = reference_moments(
        _references(reference_in, len(delta), "reference_in"),
        _references(reference_out, len(delta), "reference_out"),
    )
    variance = np.maximum(sigma ** 2 * gradient ** 2 + tau2, _MIN_VARIANCE)
    return ((delta - mean_out) ** 2 - (delta - mean_in) ** 2) / (2.0 * variance)


def increment_scores(*args, **kwargs) -> np.ndarray:
    """Cumulative increment log-likelihood ratio: score at round r sums rounds 1..r."""
    return np.cumsum(increment_llr(*args, **kwargs))


def roc_auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Probability that a random IN score exceeds a random OUT score (ties count 1/2)."""
    positive, negative = np.asarray(positive, dtype=float), np.asarray(negative, dtype=float)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("need at least one positive and one negative score")
    scores = np.concatenate([positive, negative])
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    # average ranks over ties
    boundaries = np.flatnonzero(np.diff(sorted_scores)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(scores)]])
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (start + end + 1) / 2.0
    rank_sum = ranks[: positive.size].sum()
    return float((rank_sum - positive.size * (positive.size + 1) / 2.0) / (positive.size * negative.size))


def tpr_at_fpr(positive: Sequence[float], negative: Sequence[float], fpr: float) -> float:
    """Largest TPR of a rule ``score >= threshold`` whose FPR does not exceed ``fpr``.

    Thresholds are the observed scores; ties are flagged together, so a tied block of
    negatives can only be crossed as a whole (conservative).
    """
    if not 0.0 <= fpr < 1.0:
        raise ValueError("fpr must be in [0, 1)")
    positive, negative = np.asarray(positive, dtype=float), np.asarray(negative, dtype=float)
    if positive.size == 0 or negative.size == 0:
        raise ValueError("need at least one positive and one negative score")
    best = 0.0
    for threshold in np.unique(np.concatenate([positive, negative, [np.inf]])):
        if np.mean(negative >= threshold) <= fpr:
            best = max(best, float(np.mean(positive >= threshold)))
    return best


def stratified_cluster_bootstrap(
    strata: Sequence[Sequence[object]],
    statistic: Callable[[list[list[object]]], float],
    *,
    resamples: int = 2000,
    seed: int = 1729,
) -> tuple[float, tuple[float, float]]:
    """Point estimate and 95% percentile interval, resampling whole clusters within strata.

    A cluster is one target-seed pair (all rounds of its IN and OUT trajectories, so
    within-pair correlation is kept); a stratum is one analysis cell. ``statistic``
    receives, per stratum, the list of selected clusters.
    """
    if not strata or any(len(stratum) == 0 for stratum in strata):
        raise ValueError("every stratum needs at least one cluster")
    rng = np.random.default_rng(seed)
    point = statistic([list(stratum) for stratum in strata])
    draws = []
    for _ in range(resamples):
        draws.append(statistic([
            [stratum[i] for i in rng.integers(0, len(stratum), len(stratum))] for stratum in strata
        ]))
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(point), (float(low), float(high))


def pooled(metric: Callable[[np.ndarray, np.ndarray], float]) -> Callable[[list[list[object]]], float]:
    """Statistic for clusters ``(positive_scores, negative_scores)``, pooled over all strata."""

    def statistic(selected: list[list[object]]) -> float:
        clusters = [cluster for stratum in selected for cluster in stratum]
        return metric(np.concatenate([c[0] for c in clusters]),
                      np.concatenate([c[1] for c in clusters]))

    return statistic
