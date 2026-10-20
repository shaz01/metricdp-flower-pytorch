"""M0 (CIA roadmap v2): server noise drawn from a dedicated generator seeded by ``noise-seed``.

Without a noise seed every strategy keeps drawing from NumPy's global state, so committed
runs replay unchanged. With one, the draws depend on the noise seed only (not on the global
state that ``seed_training`` sets), which is what makes two worlds' noise independent.
"""

from __future__ import annotations

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord
from flwr.serverapp.strategy import FedAvg

from metricdp_pytorch.private_metric_strategy import PrivateMedianMetricPrivacyServerSideFixedClipping
from metricdp_pytorch.residual_clipping_strategy import ResidualClippingServerSide
from metricdp_pytorch.seeded_noise import NOISE_BLOCKS, noise_diagnostics, set_noise_seed
from metricdp_pytorch.strategy_factory import UNIT_WEIGHT_KEY, make_strategy

SHAPE = (64, 32)


def zeros():
    return ArrayRecord({"w": Array(np.zeros(SHAPE, dtype=np.float32)), "b": Array(np.zeros(7, dtype=np.float32))})


def flat(record):
    return np.concatenate([a.ravel() for a in record.to_numpy_ndarrays()])


def make(privacy):
    if privacy == "private-median":
        return PrivateMedianMetricPrivacyServerSideFixedClipping(
            FedAvg(weighted_by_key=UNIT_WEIGHT_KEY), noise_multiplier=0.5, clipping_norm=5.0, num_slots=4,
            initial_quantile=1.0, minimum_quantile=0.1, maximum_quantile=10.0, seed=0)
    strategy = make_strategy("fedavg", privacy, num_clients=4, fraction_evaluate=1.0,
                             noise_multiplier=0.5, clipping_norm=5.0)
    if privacy == "metric-privacy":
        strategy.current_distance = 0.37
    return strategy


def draw(privacy, noise_seed=None, global_seed=0):
    strategy = make(privacy)
    if noise_seed is not None:
        set_noise_seed(strategy, noise_seed)
    strategy.current_arrays = zeros()
    if privacy == "private-median":
        strategy._active_clients = 4
    np.random.seed(global_seed)
    return flat(strategy._add_noise_to_aggregated_arrays(zeros()))


PRIVACIES = ["global-dp", "metric-privacy", "private-median"]


@pytest.mark.parametrize("privacy", PRIVACIES)
def test_noise_seed_fixes_the_draw_whatever_the_global_state(privacy):
    np.testing.assert_array_equal(draw(privacy, 7, global_seed=1), draw(privacy, 7, global_seed=2))


@pytest.mark.parametrize("privacy", PRIVACIES)
def test_different_noise_seeds_draw_uncorrelated_noise(privacy):
    a, b = draw(privacy, 7), draw(privacy, 8)
    assert abs(np.corrcoef(a, b)[0, 1]) < 0.1


@pytest.mark.parametrize("privacy", PRIVACIES)
def test_without_noise_seed_the_global_state_is_used(privacy):
    """Legacy path: same global state, same noise (the v1 coupling); different state, different noise."""
    np.testing.assert_array_equal(draw(privacy, global_seed=3), draw(privacy, global_seed=3))
    assert not np.array_equal(draw(privacy, global_seed=3), draw(privacy, global_seed=4))


@pytest.mark.parametrize("privacy", PRIVACIES)
def test_seeded_noise_has_the_configured_std(privacy):
    strategy = make(privacy)
    set_noise_seed(strategy, 11)
    strategy.current_arrays = zeros()
    if privacy == "private-median":
        strategy._active_clients = 4
    noise = flat(strategy._add_noise_to_aggregated_arrays(zeros()))
    expected = strategy._noise_stdv if privacy == "private-median" else strategy.current_noise_stdv
    assert noise.std() == pytest.approx(expected, rel=0.05)


def test_residual_clipping_noise_seed_replaces_its_seed():
    def run(seed, noise_seed):
        strategy = ResidualClippingServerSide(FedAvg(weighted_by_key=UNIT_WEIGHT_KEY), noise_multiplier=0.5,
                                              num_slots=4, rho=1.0, seed=seed)
        if noise_seed is not None:
            set_noise_seed(strategy, noise_seed)
        return strategy.rng.normal(size=100)

    np.testing.assert_array_equal(run(0, 5), run(1, 5))
    np.testing.assert_array_equal(run(0, None), np.random.default_rng(0).normal(size=100))


def test_noise_diagnostics_match_a_hand_computation():
    rng = np.random.default_rng(0)
    current = [rng.normal(size=(10, 30)), rng.normal(size=5)]
    signal = [rng.normal(size=(10, 30)) * 0.1, rng.normal(size=5) * 0.1]
    noise = [rng.normal(size=(10, 30)) * 0.2, rng.normal(size=5) * 0.2]
    aggregated = [c + s for c, s in zip(current, signal)]
    released = [a + n for a, n in zip(aggregated, noise)]
    diagnostics = noise_diagnostics(aggregated, released, current, std=0.2)

    delta = np.concatenate([(r - c).ravel() for r, c in zip(released, current)])
    assert diagnostics["dp-released-update-norm"] == pytest.approx(np.linalg.norm(delta))
    blocks = diagnostics["dp-noise-energy-blocks"]
    assert len(blocks) == NOISE_BLOCKS
    s = np.concatenate([x.ravel() for x in signal])
    xi = np.concatenate([x.ravel() for x in noise])
    for block, d_b, s_b, xi_b in zip(blocks, np.array_split(delta, NOISE_BLOCKS), np.array_split(s, NOISE_BLOCKS),
                                     np.array_split(xi, NOISE_BLOCKS)):
        assert block == pytest.approx(d_b @ d_b - s_b @ s_b - 0.04 * len(d_b))
        assert block == pytest.approx(xi_b @ xi_b + 2 * s_b @ xi_b - 0.04 * len(d_b))


@pytest.mark.parametrize("privacy", ["global-dp", "metric-privacy", "private-median"])
def test_strategies_log_noise_diagnostics_only_when_asked(privacy):
    strategy = make(privacy)
    set_noise_seed(strategy, 3)
    strategy.current_arrays = zeros()
    if privacy == "private-median":
        strategy._active_clients = 4
    strategy._add_noise_to_aggregated_arrays(zeros())
    assert "dp-noise-energy-blocks" not in strategy.noise_round_diagnostics

    strategy.log_noise_diagnostics = True
    released = strategy._add_noise_to_aggregated_arrays(zeros())
    assert strategy.noise_round_diagnostics["dp-released-update-norm"] == pytest.approx(
        np.linalg.norm(flat(released)), rel=1e-5)
    assert len(strategy.noise_round_diagnostics["dp-noise-energy-blocks"]) == NOISE_BLOCKS
