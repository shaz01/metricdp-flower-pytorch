"""D3 crowd bound at head dimensions (roadmap v2 B4; the D6 question).

Propositions 1 and 2 of ``research/cia_review_2026-10/theory/f_cmip.md``, evaluated per round on
the clipped client updates of a head trajectory. Round release ``R = theta + (1/n) sum_j u_j + Z``,
``Z ~ N(0, sigma^2 I)``, ``sigma = z C / n``; the crowd ``B = sum_{j != i} u_j`` is modelled as
``N(m_B, (n - 1) Sigma_pop)`` with ``Sigma_pop`` the sample covariance of the other clients' updates
in that round (rank at most ``n - 2``), so ``S = sigma^2 I + (n - 1) Sigma_pop / n^2``.

* Full release (Proposition 1): ``mu_c = ||S^{-1/2} u_i|| / n``; ratio to the round's GDP value
  ``C / (n sigma)`` is ``sigma ||S^{-1/2} u_i|| / C``.
* Statistic ``w^T R`` (Proposition 2, ``k = 1``): ``rho = sigma ||w|| / sqrt(sigma^2 ||w||^2 + w^T Sigma_B w / n^2)``.

``u^T S^{-1} u`` is computed with the Woodbury identity on the ``d x (n - 1)`` factor, so
``d = 5,130`` costs nothing. Rounds compose under independent crowd terms (Proposition 3.1) as
``sqrt(mean_r ratio_r^2)``; Proposition 3.2 shows that without that assumption the worst case is GDP.
"""

from __future__ import annotations

import numpy as np


def _crowd_factor(others: np.ndarray, num_slots: int) -> np.ndarray:
    """``A`` with ``Sigma_B / n^2 = A A^T`` from the other clients' updates (rows)."""
    count = len(others)
    if count < 2:
        raise ValueError("Need at least two other clients to estimate the crowd covariance.")
    centred = others - others.mean(axis=0)
    scale = np.sqrt(count / (count - 1)) / num_slots  # (n - 1) Sigma_pop / n^2, Sigma_pop unbiased
    return (centred * scale).T


def full_release_ratio(updates: np.ndarray, target: int, sigma: float, clip: float, num_slots: int) -> float:
    """``mu_c / mu_GDP`` for one round (Proposition 1, worst-case GDP signal ``C``)."""
    u = updates[target]
    factor = _crowd_factor(np.delete(updates, target, axis=0), num_slots)
    projected = factor.T @ u
    inner = sigma ** 2 * np.eye(factor.shape[1]) + factor.T @ factor
    quadratic = (u @ u - projected @ np.linalg.solve(inner, projected)) / sigma ** 2  # u^T S^{-1} u
    return float(sigma * np.sqrt(max(quadratic, 0.0)) / clip)


def statistic_ratio(updates: np.ndarray, target: int, direction: np.ndarray, sigma: float, num_slots: int) -> float:
    """``rho`` for the one-dimensional statistic ``direction^T R`` (Proposition 2)."""
    factor = _crowd_factor(np.delete(updates, target, axis=0), num_slots)
    noise = sigma ** 2 * float(direction @ direction)
    crowd = float(np.sum((factor.T @ direction) ** 2))
    return float(np.sqrt(noise / (noise + crowd)))


def compose(ratios: np.ndarray) -> float:
    """Composition over rounds with independent crowd terms (Proposition 3.1)."""
    return float(np.sqrt(np.mean(np.square(ratios))))
