"""Accountant for client-level DP-FedAvg (roadmap v2, Agent C; used by B1 and C1).

Closed-form tests run everywhere. Tests that need Google's ``dp-accounting`` are skipped unless
it is importable: ``uv run --with dp-accounting==0.6.0 pytest tests/test_accounting.py``.

Published values (each checked against its source on 2026-10-17):

* Abadi et al. 2016, Section 5.1: q = 0.01, sigma = 4, delta = 1e-5; moments accountant
  epsilon 1.26 at E = 100 epochs (T = 10,000 steps) and 2.55 at E = 400 (T = 40,000).
* TensorFlow Privacy, ``tutorials/README.md`` (MNIST, N = 60,000, batch 256, delta = 1e-5, RDP
  accountant): noise 1.1 for 60 epochs gives epsilon 3.01, noise 1.3 for 15 epochs 1.19, noise
  0.7 for 45 epochs 7.10. Steps are ``ceil(epochs * N / batch)`` as in ``compute_dp_sgd_privacy``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import integrate
from scipy.stats import norm

from metricdp_pytorch import accounting as acc
from metricdp_pytorch.accounting import SubsampledGaussian

MNIST_N, MNIST_BATCH = 60_000, 256


def _requires_dp_accounting():
    return pytest.importorskip("dp_accounting")


def _hockey_stick(epsilon: float, mu: float) -> float:
    """``sup_A P(A) - e^eps Q(A)`` for ``P = N(mu, 1)``, ``Q = N(0, 1)``, by quadrature."""
    def integrand(x):
        return max(norm.pdf(x, loc=mu) - math.exp(epsilon) * norm.pdf(x), 0.0)
    # The integrand is positive exactly where x > eps/mu + mu/2.
    start = epsilon / mu + mu / 2
    value, _ = integrate.quad(integrand, start, start + 40, limit=200, epsabs=0.0, epsrel=1e-10)
    return value


# --- Closed forms ---------------------------------------------------------------------------------

@pytest.mark.parametrize("mu", [0.3, 1.0, 2.5, 8.0])
@pytest.mark.parametrize("epsilon", [0.0, 0.5, 2.0, 6.0])
def test_gaussian_delta_is_the_hockey_stick_divergence(mu, epsilon):
    expected = _hockey_stick(epsilon, mu)
    assert acc.gaussian_delta(epsilon, mu) == pytest.approx(expected, rel=1e-6, abs=1e-14)


def test_gaussian_delta_is_stable_for_large_epsilon():
    values = acc.gaussian_delta(np.array([20.0, 50.0, 200.0, 800.0]), 3.0)
    assert np.all(np.isfinite(values)) and np.all(values >= 0)
    assert values[1] == pytest.approx(norm.cdf(-50 / 3 + 1.5), rel=1e-6)  # second term negligible
    assert np.all(np.diff(values) <= 0)


@pytest.mark.parametrize("mu", [0.5, 1.0, 4.0, 20.0])
@pytest.mark.parametrize("delta", [1e-3, 1e-5, 1e-9])
def test_gdp_epsilon_inverts_gaussian_delta(mu, delta):
    eps = acc.gdp_epsilon(mu, delta)
    assert acc.gaussian_delta(eps, mu) == pytest.approx(delta, rel=1e-6)


def test_gdp_epsilon_is_zero_when_total_variation_is_below_delta():
    mu = 1e-6  # TV distance 2 Phi(mu/2) - 1, about 4e-7
    assert acc.gdp_epsilon(mu, 1e-5) == 0.0


def test_full_participation_mu_composes_in_quadrature():
    # Dong, Roth and Su 2022, Corollary 3.3: T rounds of (1/z)-GDP are (sqrt(T)/z)-GDP.
    assert acc.gdp_mu(4.0, 16) == pytest.approx(1.0)
    report = acc.dominating_mu(SubsampledGaussian(noise_multiplier=2.0, sampling_rate=1.0, rounds=9))
    assert report.mu == pytest.approx(1.5)


def test_noise_for_mu_is_closed_form_at_full_participation():
    assert acc.noise_multiplier_for_mu(2.0, rounds=20) == pytest.approx(math.sqrt(20) / 2.0)


def test_mu_clt_formula():
    # Bu, Dong, Long and Su 2020, Eq. (5).
    assert acc.mu_clt(1.0, 0.01, 10_000) == pytest.approx(0.01 * math.sqrt(10_000 * (math.e - 1)))


@pytest.mark.parametrize("kwargs", [
    {"noise_multiplier": 0.0}, {"noise_multiplier": 1.0, "sampling_rate": 0.0},
    {"noise_multiplier": 1.0, "sampling_rate": 1.5}, {"noise_multiplier": 1.0, "rounds": 0},
])
def test_mechanism_validation(kwargs):
    with pytest.raises(ValueError):
        SubsampledGaussian(**kwargs)


# --- Library-backed: published values -------------------------------------------------------------

@pytest.mark.parametrize("sigma, epochs, published", [(1.1, 60, 3.01), (1.3, 15, 1.19), (0.7, 45, 7.10)])
def test_reproduces_tensorflow_privacy_mnist_table(sigma, epochs, published):
    _requires_dp_accounting()
    steps = math.ceil(epochs * MNIST_N / MNIST_BATCH)
    mechanism = SubsampledGaussian(sigma, MNIST_BATCH / MNIST_N, steps)
    assert round(acc.epsilon(mechanism, 1e-5, method="rdp-classical"), 2) == published


def test_reproduces_abadi_moments_accountant_example():
    _requires_dp_accounting()
    mechanism = SubsampledGaussian(4.0, 0.01, 10_000)
    assert round(acc.epsilon(mechanism, 1e-5, method="rdp-classical"), 2) == 1.26


def test_abadi_400_epoch_value_within_published_rounding_of_their_implementation():
    # Abadi et al. report 2.55 at E = 400; the RDP orders of TensorFlow Privacy give 2.574. The
    # moments accountant evaluated its bound at integer orders only (lambda <= 32), so the two
    # differ in the second decimal. Both bound the same mechanism.
    _requires_dp_accounting()
    mechanism = SubsampledGaussian(4.0, 0.01, 40_000)
    assert acc.epsilon(mechanism, 1e-5, method="rdp-classical") == pytest.approx(2.55, abs=0.03)


@pytest.mark.parametrize("mechanism", [
    SubsampledGaussian(4.0, 0.01, 10_000),
    SubsampledGaussian(1.1, MNIST_BATCH / MNIST_N, math.ceil(60 * MNIST_N / MNIST_BATCH)),
    SubsampledGaussian(1.0, 0.25, 50),
    SubsampledGaussian(5.0, 0.5, 100),
])
def test_pld_is_tighter_than_rdp(mechanism):
    _requires_dp_accounting()
    pld = acc.epsilon(mechanism, 1e-5, method="pld")
    rdp = acc.epsilon(mechanism, 1e-5, method="rdp")
    classical = acc.epsilon(mechanism, 1e-5, method="rdp-classical")
    assert pld <= rdp <= classical + 1e-9
    assert pld > 0.5 * rdp  # same mechanism: the bounds are not wildly apart


# --- Library-backed: PLD at full participation equals exact GDP -----------------------------------

@pytest.mark.parametrize("z, rounds", [(2.0, 16), (4.0, 10), (0.5, 1)])
def test_pld_at_full_participation_matches_exact_gdp(z, rounds):
    _requires_dp_accounting()
    mechanism = SubsampledGaussian(z, 1.0, rounds)
    pld = acc.privacy_loss_distribution(mechanism)
    exact = acc.gdp_epsilon(acc.gdp_mu(z, rounds), 1e-5)
    library = float(pld.get_epsilon_for_delta(1e-5))
    assert exact <= library <= exact + 0.01  # pessimistic discretization: never optimistic


# --- Library-backed: dominating mu ----------------------------------------------------------------

@pytest.mark.parametrize("mechanism", [
    SubsampledGaussian(1.0, 0.25, 50),
    SubsampledGaussian(3.0, 0.5, 20),
    SubsampledGaussian(0.8, 0.05, 200),
])
def test_dominating_mu_dominates_the_privacy_profile(mechanism):
    _requires_dp_accounting()
    result = acc.dominating_mu(mechanism)
    assert result.mu_lower <= result.mu <= result.mu_lower * (1 + 1e-3)
    pld = acc.privacy_loss_distribution(mechanism)
    grid = np.sort(np.random.default_rng(0).uniform(0.0, 1.5 * result.eps_max, 2000))
    profile = np.asarray(pld.get_delta_for_epsilon(grid))
    assert np.all(profile <= np.maximum(acc.gaussian_delta(grid, result.mu), result.delta_floor))
    # Tight: one percent below the pointwise lower bound fails at some epsilon.
    finer = np.linspace(0.0, result.eps_max, result.grid_size + 1)
    profile = np.asarray(pld.get_delta_for_epsilon(finer))
    assert np.any(profile > np.maximum(acc.gaussian_delta(finer, 0.99 * result.mu_lower), result.delta_floor))


def test_sampling_amplifies_and_is_monotone():
    _requires_dp_accounting()
    z, rounds = 2.0, 50
    mus = [acc.dominating_mu(SubsampledGaussian(z, q, rounds)).mu for q in (0.1, 0.25, 0.5)]
    assert mus[0] < mus[1] < mus[2] < acc.gdp_mu(z, rounds)
    noisier = acc.dominating_mu(SubsampledGaussian(2 * z, 0.25, rounds)).mu
    longer = acc.dominating_mu(SubsampledGaussian(z, 0.25, 2 * rounds)).mu
    assert noisier < mus[1] < longer


def test_dominating_mu_approaches_the_clt_in_its_regime():
    # Small q, many rounds, q sqrt(T) fixed: Bu et al. 2020, Theorem 5.
    _requires_dp_accounting()
    mechanism = SubsampledGaussian(1.0, 0.01, 10_000)
    result = acc.dominating_mu(mechanism)
    clt = acc.mu_clt(1.0, 0.01, 10_000)
    assert clt <= result.mu <= 1.10 * clt


def test_mu_clt_is_not_a_bound_outside_its_regime():
    # Large q and few rounds: the CLT overstates mu here, so it is never used as a guarantee.
    _requires_dp_accounting()
    mechanism = SubsampledGaussian(0.8, 0.5, 50)
    assert acc.dominating_mu(mechanism).mu < acc.mu_clt(0.8, 0.5, 50)


def test_dominating_mu_grows_as_the_floor_falls():
    _requires_dp_accounting()
    mechanism = SubsampledGaussian(2.0, 0.25, 50)
    mus = [acc.dominating_mu(mechanism, delta_floor=floor).mu for floor in (1e-4, 1e-7, 1e-10)]
    assert mus[0] < mus[1] < mus[2] < acc.gdp_mu(2.0, 50)


def test_noise_for_mu_round_trip_with_sampling():
    _requires_dp_accounting()
    z = acc.noise_multiplier_for_mu(1.0, sampling_rate=0.25, rounds=50, rtol=1e-2)
    assert acc.dominating_mu(SubsampledGaussian(z, 0.25, 50)).mu <= 1.0
    assert acc.dominating_mu(SubsampledGaussian(z / 1.02, 0.25, 50)).mu > 1.0


def test_noise_for_epsilon_round_trip():
    _requires_dp_accounting()
    z = acc.noise_multiplier_for_epsilon(2.0, 1e-5, sampling_rate=0.5, rounds=20, rtol=1e-2)
    assert acc.epsilon(SubsampledGaussian(z, 0.5, 20), 1e-5) <= 2.0
    assert acc.epsilon(SubsampledGaussian(z / 1.02, 0.5, 20), 1e-5) > 2.0


def test_report_has_every_quantity():
    _requires_dp_accounting()
    out = acc.report(SubsampledGaussian(1.0, 0.25, 50))
    assert out["mu_dominating"] < out["mu_full_participation"]
    assert out["epsilon_pld"] <= out["epsilon_rdp"]
    assert out["mu_dominating_lower"] <= out["mu_dominating"]
    assert set(out) >= {"mu_clt", "mu_dominating_delta_floor", "delta", "rounds"}
