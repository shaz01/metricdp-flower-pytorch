"""Client-level privacy accounting for DP-FedAvg with Gaussian noise and Poisson client sampling.

One round releases ``sum_{j in S} clip(u_j, C) + N(0, (z C)^2 I)`` divided by a public
denominator, where every client joins ``S`` independently with probability ``q`` (``q = 1``:
full participation). Under add/remove-one-client adjacency this is the Poisson-subsampled
Gaussian mechanism with noise multiplier ``z``; ``T`` rounds compose it ``T`` times.

* ``q = 1`` is exact GDP: ``mu = sqrt(T) / z`` (Dong, Roth and Su 2022, Corollary 3.3).
* ``q < 1`` uses privacy loss distributions from Google's ``dp-accounting`` library
  (pessimistic discretization, so ``delta`` and ``epsilon`` are upper bounds; the maximum of
  the add and remove directions).
* ``dominating_mu`` is the smallest ``mu`` for which the mechanism is ``mu``-GDP up to an explicit
  ``delta_floor``: its privacy profile satisfies ``delta(eps) <= max(delta_mu(eps), delta_floor)``
  for every ``eps >= 0`` (Dong, Roth and Su 2022, Corollary 2.13, with the floor added). This is
  the "amplified formal mu" of roadmap v2 B1; report it with its floor.
* ``mu_clt`` is the central-limit approximation ``q sqrt(T (exp(1/z^2) - 1))`` for Poisson
  sampling (Bu, Dong, Long and Su 2020, Eq. (5) and Theorem 5): asymptotic, not a guarantee.

``dp-accounting`` is imported lazily, so this module loads without it; the project venv does not
ship it. Run code that uses PLD or RDP through ``uv run --with dp-accounting==0.6.0``.
Definitions and checks: ``research/cia_roadmap_v2/theory/accounting.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

DP_ACCOUNTING_VERSION = "0.6.0"
DEFAULT_DISCRETIZATION = 1e-4
# Orders of TensorFlow Privacy's ``compute_dp_sgd_privacy`` (the published DP-SGD examples).
RDP_ORDERS = tuple([1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)))


def _dp_accounting() -> Any:
    try:
        import dp_accounting
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "metricdp_pytorch.accounting needs Google's dp-accounting for q < 1 or RDP; "
            f"run with `uv run --with dp-accounting=={DP_ACCOUNTING_VERSION}`."
        ) from error
    return dp_accounting


@dataclass(frozen=True)
class SubsampledGaussian:
    """``rounds`` compositions of the Poisson-subsampled Gaussian mechanism."""

    noise_multiplier: float
    sampling_rate: float = 1.0
    rounds: int = 1

    def __post_init__(self) -> None:
        if not self.noise_multiplier > 0:
            raise ValueError("noise_multiplier must be positive.")
        if not 0 < self.sampling_rate <= 1:
            raise ValueError("sampling_rate must be in (0, 1].")
        if self.rounds < 1 or int(self.rounds) != self.rounds:
            raise ValueError("rounds must be a positive integer.")

    @property
    def full_participation(self) -> bool:
        return self.sampling_rate == 1.0


# --- Gaussian differential privacy (closed forms) -------------------------------------------------

def gaussian_delta(epsilon: float | np.ndarray, mu: float) -> float | np.ndarray:
    """Privacy profile of ``mu``-GDP: ``Phi(-eps/mu + mu/2) - e^eps Phi(-eps/mu - mu/2)``.

    Dong, Roth and Su 2022, Corollary 2.13; equal to the analytic Gaussian mechanism of Balle and
    Wang 2018 (Theorem 8) with ``mu = sensitivity / sigma``. Computed in log space so that large
    ``eps`` does not overflow.
    """
    eps = np.asarray(epsilon, dtype=float)
    if mu <= 0:
        return np.zeros_like(eps) if eps.ndim else 0.0
    value = _gaussian_delta_array(eps, np.asarray(mu, dtype=float))
    return value if eps.ndim else float(value)


def gdp_epsilon(mu: float, delta: float) -> float:
    """Smallest ``epsilon >= 0`` with ``gaussian_delta(epsilon, mu) <= delta``."""
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    if gaussian_delta(0.0, mu) <= delta:
        return 0.0
    upper = mu * mu / 2 + mu * (abs(norm.ppf(delta)) + 1) + 1
    return float(brentq(lambda e: gaussian_delta(e, mu) - delta, 0.0, upper, xtol=1e-12))


def gdp_mu(noise_multiplier: float, rounds: int) -> float:
    """``mu`` of ``rounds`` full-participation rounds: ``sqrt(T) / z`` (exact)."""
    return math.sqrt(rounds) / noise_multiplier


def mu_clt(noise_multiplier: float, sampling_rate: float, rounds: int) -> float:
    """Central-limit ``mu`` of the Poisson-subsampled Gaussian (Bu et al. 2020, Eq. 5); not a bound."""
    return sampling_rate * math.sqrt(rounds * math.expm1(noise_multiplier ** -2))


def _mu_for_delta(epsilon: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """The ``mu`` with ``gaussian_delta(epsilon, mu) = delta``, elementwise (bisection on ``log mu``).

    ``gaussian_delta`` increases in ``mu``; the result is an upper bracket, accurate to ``1e-10``
    relative. ``delta <= 0`` gives 0 and ``delta >= 1`` gives infinity.
    """
    eps = np.asarray(epsilon, dtype=float)
    target = np.broadcast_to(np.asarray(delta, dtype=float), eps.shape)
    low, high = np.full(eps.shape, math.log(1e-8)), np.full(eps.shape, math.log(1e4))
    for _ in range(80):
        middle = (low + high) / 2
        too_private = _gaussian_delta_array(eps, np.exp(middle)) < target
        low, high = np.where(too_private, middle, low), np.where(too_private, high, middle)
    mu = np.exp(high)
    return np.where(target <= 0, 0.0, np.where(target >= 1, np.inf, mu))


def _gaussian_delta_array(epsilon: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """``gaussian_delta`` for arrays of ``epsilon`` and positive ``mu`` (log space)."""
    first = norm.logcdf(-epsilon / mu + mu / 2)
    second = epsilon + norm.logcdf(-epsilon / mu - mu / 2)
    return np.exp(first) * -np.expm1(np.minimum(second - first, 0.0))


# --- Privacy loss distributions and RDP (dp-accounting) -------------------------------------------

def privacy_loss_distribution(mechanism: SubsampledGaussian, *,
                              discretization: float = DEFAULT_DISCRETIZATION) -> Any:
    """Composed PLD of ``mechanism`` (add/remove-one, pessimistic estimate)."""
    pld = _dp_accounting().pld.privacy_loss_distribution
    single = pld.from_gaussian_mechanism(
        standard_deviation=mechanism.noise_multiplier,
        sensitivity=1.0,
        sampling_prob=mechanism.sampling_rate,
        value_discretization_interval=discretization,
        pessimistic_estimate=True,
    )
    return single.self_compose(mechanism.rounds)


def _rdp_accountant(mechanism: SubsampledGaussian, orders: tuple[float, ...]) -> Any:
    dpa = _dp_accounting()
    accountant = dpa.rdp.RdpAccountant(list(orders))
    accountant.compose(dpa.SelfComposedDpEvent(
        dpa.PoissonSampledDpEvent(mechanism.sampling_rate, dpa.GaussianDpEvent(mechanism.noise_multiplier)),
        mechanism.rounds,
    ))
    return accountant


def epsilon(mechanism: SubsampledGaussian, delta: float, *, method: str = "pld",
            discretization: float = DEFAULT_DISCRETIZATION, orders: tuple[float, ...] = RDP_ORDERS) -> float:
    """``epsilon`` at ``delta``.

    ``method``: ``"pld"`` (tight up to discretization; exact GDP at ``q = 1``), ``"rdp"`` (the
    library's RDP accountant and its improved conversion) or ``"rdp-classical"`` (RDP with the
    conversion ``min_a rdp(a) + log(1/delta) / (a - 1)`` of Mironov 2017, Proposition 3, which the
    moments accountant of Abadi et al. 2016 and TensorFlow Privacy's published examples use).
    """
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1).")
    if method == "pld":
        if mechanism.full_participation:
            return gdp_epsilon(gdp_mu(mechanism.noise_multiplier, mechanism.rounds), delta)
        return float(privacy_loss_distribution(mechanism, discretization=discretization).get_epsilon_for_delta(delta))
    if method == "rdp":
        return float(_rdp_accountant(mechanism, orders).get_epsilon(delta))
    if method == "rdp-classical":
        accountant = _rdp_accountant(mechanism, orders)
        order = np.asarray(accountant.orders, dtype=float)
        values = np.asarray(accountant.rdp, dtype=float) + math.log(1 / delta) / (order - 1)
        return float(np.nanmin(values))
    raise ValueError("method must be 'pld', 'rdp' or 'rdp-classical'.")


def delta(mechanism: SubsampledGaussian, epsilon_value: float, *,
          discretization: float = DEFAULT_DISCRETIZATION) -> float:
    """``delta`` at ``epsilon_value`` (PLD; exact GDP at ``q = 1``)."""
    if mechanism.full_participation:
        return float(gaussian_delta(epsilon_value, gdp_mu(mechanism.noise_multiplier, mechanism.rounds)))
    return float(privacy_loss_distribution(mechanism, discretization=discretization)
                 .get_delta_for_epsilon(epsilon_value))


@dataclass(frozen=True)
class DominatingMu:
    """Result of ``dominating_mu``.

    The mechanism satisfies ``delta(eps) <= max(delta_mu(eps), delta_floor)`` for every ``eps >= 0``
    with ``mu`` = ``mu`` (a rigorous upper bound); ``mu_lower`` is the largest ``mu`` needed at the
    grid points themselves, so the smallest valid ``mu`` lies in ``[mu_lower, mu]``.
    """

    mu: float
    mu_lower: float
    delta_floor: float
    eps_max: float
    binding_epsilon: float
    grid_size: int


def dominating_mu(mechanism: SubsampledGaussian, *, delta_floor: float = 1e-10, rtol: float = 1e-3,
                  discretization: float = DEFAULT_DISCRETIZATION, max_grid_size: int = 2 ** 20) -> DominatingMu:
    """Smallest ``mu`` with ``delta(eps) <= max(gaussian_delta(eps, mu), delta_floor)`` for all ``eps``.

    "``mu``-GDP up to ``delta_floor``": the subsampled Gaussian is not exactly GDP, and the ``mu``
    needed at ``delta`` far below any meaningful level is driven by the tail of its privacy loss,
    so the floor is explicit. With ``delta_floor = 0`` this is ``mu``-GDP (Dong, Roth and Su 2022,
    Corollary 2.13).

    Method: on a uniform grid ``0 = e_0 < ... < e_K = eps_max`` (``eps_max`` the first point where
    the PLD's pessimistic ``delta`` is below ``delta_floor``), require ``delta(e_k) <=
    delta_mu(e_{k+1})`` whenever ``delta(e_k) > delta_floor``. Both profiles decrease in ``eps``,
    so this implies the claim on every ``[e_k, e_{k+1}]``. The grid is refined until this upper
    bound is within ``rtol`` of the pointwise lower bound. At ``q = 1`` the exact ``sqrt(T) / z``.
    """
    if not 0 <= delta_floor < 1:
        raise ValueError("delta_floor must be in [0, 1).")
    if mechanism.full_participation:
        mu = gdp_mu(mechanism.noise_multiplier, mechanism.rounds)
        return DominatingMu(mu=mu, mu_lower=mu, delta_floor=delta_floor, eps_max=math.inf,
                            binding_epsilon=math.nan, grid_size=0)
    if delta_floor == 0:
        raise ValueError("delta_floor = 0 needs the privacy loss's full tail; use a positive floor.")
    pld = privacy_loss_distribution(mechanism, discretization=discretization)
    eps_max = 1.0
    while float(pld.get_delta_for_epsilon(eps_max)) > delta_floor:
        eps_max *= 2
    size = 1024
    while True:
        grid = np.linspace(0.0, eps_max, size + 1)
        deltas = np.asarray(pld.get_delta_for_epsilon(grid), dtype=float)
        active = deltas[:-1] > delta_floor
        upper = _mu_for_delta(grid[1:][active], deltas[:-1][active])
        lower = _mu_for_delta(grid[deltas > delta_floor], deltas[deltas > delta_floor])
        k = int(np.argmax(upper))
        mu, mu_lower = float(upper[k]), float(lower.max())
        if mu <= mu_lower * (1 + rtol) or size >= max_grid_size:
            return DominatingMu(mu=mu, mu_lower=mu_lower, delta_floor=delta_floor, eps_max=float(eps_max),
                                binding_epsilon=float(grid[:-1][active][k]), grid_size=size)
        size *= 2


# --- Calibration -----------------------------------------------------------------------------------

def noise_multiplier_for_mu(target_mu: float, *, sampling_rate: float = 1.0, rounds: int = 1,
                            rtol: float = 1e-3, **kwargs: Any) -> float:
    """Smallest noise multiplier whose ``dominating_mu`` is at most ``target_mu`` (to ``rtol``)."""
    if target_mu <= 0:
        raise ValueError("target_mu must be positive.")
    if sampling_rate == 1.0:
        return math.sqrt(rounds) / target_mu
    return _bisect_noise(lambda z: dominating_mu(SubsampledGaussian(z, sampling_rate, rounds), **kwargs).mu,
                         target_mu, rtol)


def noise_multiplier_for_epsilon(target_epsilon: float, delta_value: float, *, sampling_rate: float = 1.0,
                                 rounds: int = 1, rtol: float = 1e-3, **kwargs: Any) -> float:
    """Smallest noise multiplier with PLD ``epsilon(delta_value)`` at most ``target_epsilon``."""
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be positive.")
    return _bisect_noise(lambda z: epsilon(SubsampledGaussian(z, sampling_rate, rounds), delta_value, **kwargs),
                         target_epsilon, rtol)


def _bisect_noise(privacy_of: Any, target: float, rtol: float) -> float:
    """Geometric bisection on ``z`` for a privacy measure that decreases in ``z``."""
    low, high = 1e-2, 1.0
    while privacy_of(high) > target:
        low, high = high, high * 2
    while privacy_of(low) <= target and low > 1e-6:
        high, low = low, low / 2
    while high / low > 1 + rtol:
        middle = math.sqrt(low * high)
        if privacy_of(middle) > target:
            low = middle
        else:
            high = middle
    return high


def report(mechanism: SubsampledGaussian, *, delta_value: float = 1e-5, with_rdp: bool = True) -> dict[str, float]:
    """All quantities for one mechanism, for result JSONs and findings tables."""
    out: dict[str, float] = {
        "noise_multiplier": mechanism.noise_multiplier,
        "sampling_rate": mechanism.sampling_rate,
        "rounds": float(mechanism.rounds),
        "delta": delta_value,
        "mu_full_participation": gdp_mu(mechanism.noise_multiplier, mechanism.rounds),
        "mu_clt": mu_clt(mechanism.noise_multiplier, mechanism.sampling_rate, mechanism.rounds),
    }
    dominance = dominating_mu(mechanism)
    out["mu_dominating"] = dominance.mu
    out["mu_dominating_lower"] = dominance.mu_lower
    out["mu_dominating_delta_floor"] = dominance.delta_floor
    out["epsilon_pld"] = epsilon(mechanism, delta_value, method="pld")
    if with_rdp:
        out["epsilon_rdp"] = epsilon(mechanism, delta_value, method="rdp")
    return out
