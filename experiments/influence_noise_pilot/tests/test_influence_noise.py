"""Synthetic tests only: no dataset, no training."""
import json
from pathlib import Path

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping
from flwr.supercore.differential_privacy import compute_clip_model_update, compute_stdv

from metricdp_pytorch import influence_noise as inf
from metricdp_pytorch.strategy_factory import (
    DeterministicFedAvg, EXPERIMENTAL_PRIVACY_MODES, PRIVACY_MODES, make_base_strategy, make_strategy,
)


def rows(n=4, dim=6, seed=0):
    return np.random.default_rng(seed).normal(size=(n, dim))


# ---------------------------------------------------------------- math
@pytest.mark.parametrize("fraction", [0.0, 0.3, 0.5, 1.0])
def test_trace_matches_isotropic_energy(fraction):
    d = rows()
    sigma = inf.covariance(d, tau=0.7, fraction=fraction)
    assert np.trace(sigma) == pytest.approx(0.7**2 * d.shape[1], rel=1e-12)
    assert np.all(np.linalg.eigvalsh(sigma) >= -1e-12)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.9])
def test_monte_carlo_covariance_and_energy(fraction):
    d = rows(n=3, dim=5, seed=1)
    tau, samples = 0.3, 200_000
    rng = np.random.default_rng(7)
    xi = rng.standard_normal((samples, 5))
    eta = rng.standard_normal((samples, 3))
    trace = float(np.sum(d * d))
    z = tau * np.sqrt(1 - fraction) * xi + tau * np.sqrt(fraction * 5 / trace) * eta @ d
    empirical = z.T @ z / samples
    analytic = inf.covariance(d, tau=tau, fraction=fraction)
    assert np.max(np.abs(empirical - analytic)) < 0.02 * np.max(np.abs(analytic)) + 1e-3
    assert np.mean(np.sum(z * z, axis=1)) == pytest.approx(tau**2 * 5, rel=0.01)
    # draw_noise implements exactly this sampler
    draws = np.stack([inf.draw_noise(d, tau=tau, fraction=fraction, rng=np.random.default_rng(s)).noise
                      for s in range(20_000)])
    assert np.mean(np.sum(draws**2, axis=1)) == pytest.approx(tau**2 * 5, rel=0.03)
    assert np.max(np.abs(draws.T @ draws / len(draws) - analytic)) < 0.06 * np.max(np.abs(analytic))


def test_f0_exact_isotropic_and_common_random_numbers():
    d = rows()
    iso = inf.draw_noise(d, tau=0.2, fraction=0.0, rng=inf.round_generator(42, 3))
    rng = inf.round_generator(42, 3)
    assert np.array_equal(iso.noise, 0.2 * rng.standard_normal(d.shape[1]))
    aniso = inf.draw_noise(d, tau=0.2, fraction=0.5, rng=inf.round_generator(42, 3))
    rng = inf.round_generator(42, 3)
    xi, eta = rng.standard_normal(d.shape[1]), rng.standard_normal(d.shape[0])
    expected = 0.2 * np.sqrt(0.5) * xi + 0.2 * np.sqrt(0.5 * d.shape[1] / np.sum(d * d)) * eta @ d
    np.testing.assert_allclose(aniso.noise, expected, rtol=1e-12)


def test_degenerate_trace_falls_back_to_isotropic():
    zero = np.zeros((3, 4))
    draw = inf.draw_noise(zero, tau=0.5, fraction=0.5, rng=inf.round_generator(1, 1))
    assert draw.fallback_isotropic
    assert np.array_equal(draw.noise, 0.5 * inf.round_generator(1, 1).standard_normal(4))
    np.testing.assert_array_equal(inf.covariance(zero, tau=0.5, fraction=0.5), 0.25 * np.eye(4))


def test_seed_determinism():
    d = rows()
    a = inf.draw_noise(d, tau=1, fraction=.5, rng=inf.round_generator(42, 5)).noise
    assert np.array_equal(a, inf.draw_noise(d, tau=1, fraction=.5, rng=inf.round_generator(42, 5)).noise)
    assert not np.array_equal(a, inf.draw_noise(d, tau=1, fraction=.5, rng=inf.round_generator(42, 6)).noise)
    assert not np.array_equal(a, inf.draw_noise(d, tau=1, fraction=.5, rng=inf.round_generator(43, 5)).noise)


def test_leave_one_out_matches_explicit_renormalisation():
    u = rows(n=5, dim=7, seed=3)
    w = inf.sample_weights([10, 30, 7, 100, 3])
    g, d = inf.leave_one_out_influence(u, w)
    for i in range(5):
        keep = np.arange(5) != i
        g_minus = (w[keep] / w[keep].sum()) @ u[keep]
        np.testing.assert_allclose(d[i], g - g_minus, atol=1e-12)


def test_weights_validation():
    for bad in ([5], [1, 0], [1, float("nan")], [1, -1]):
        with pytest.raises(ValueError):
            inf.sample_weights(bad)
    assert np.all(inf.sample_weights([1, 1e6]) < 1)


def test_cap_rows():
    v = np.array([[3.0, 4.0], [0.3, 0.4], [0.0, 0.0]])
    capped, norms, flags = inf.cap_rows(v, 1.0)
    np.testing.assert_allclose(np.linalg.norm(capped, axis=1), [1.0, 0.5, 0.0])
    np.testing.assert_allclose(capped[0], [0.6, 0.8])
    np.testing.assert_array_equal(capped[1:], v[1:])
    assert norms.tolist() == [5.0, 0.5, 0.0] and flags.tolist() == [True, False, False]
    with pytest.raises(ValueError):
        inf.cap_rows(v, 0.0)


def test_clip_matches_flower_and_guards_zero():
    rng = np.random.default_rng(0)
    current = [rng.normal(size=(3, 2)).astype(np.float32), rng.normal(size=4).astype(np.float32)]
    received = [c + 10 * rng.normal(size=c.shape).astype(np.float32) for c in current]
    flower = [r.copy() for r in received]
    compute_clip_model_update(flower, current, 5.0)
    ours, norm, scale = inf.clip_update([r - c for r, c in zip(received, current)], 5.0)
    assert norm > 5 and scale < 1
    for f, c, u in zip(flower, current, ours):
        np.testing.assert_array_equal(f, c + u)
    zero, norm, scale = inf.clip_update([np.zeros(3, np.float32)], 5.0)
    assert norm == 0 and scale == 1 and not zero[0].any()


def test_directional_stdv_matches_dense_covariance():
    d = rows(n=3, dim=5, seed=4)
    trace = float(np.sum(d * d))
    sigma = inf.covariance(d, tau=0.4, fraction=0.5)
    unit = d / np.linalg.norm(d, axis=1, keepdims=True)
    expected = np.sqrt(np.einsum("id,de,ie->i", unit, sigma, unit))
    np.testing.assert_allclose(inf.directional_stdv(d, tau=0.4, fraction=0.5, trace=trace, fallback=False),
                               expected, rtol=1e-12)


# ---------------------------------------------------------------- strategy
def model(values):
    return ArrayRecord({"w": Array(np.asarray(values[:4], np.float32).reshape(2, 2)),
                        "b": Array(np.asarray(values[4:], np.float32))})


def replies(updates, counts, ids):
    out = []
    for update, count, cid in zip(updates, counts, ids):
        request = Message(content=RecordDict(), message_type="train", dst_node_id=cid + 1)
        content = RecordDict({"arrays": model(update),
                              "metrics": MetricRecord({"num-examples": count, "client-id": cid})})
        out.append(Message(content=content, reply_to=request))
    return out


def flat(record):
    return np.concatenate([a.ravel() for a in record.to_numpy_ndarrays()]).astype(np.float64)


def strategy(fraction, nm=0.0, n=4, cap=5.0):
    s = make_strategy("fedavg", "influence-noise", num_clients=n, fraction_evaluate=1.0,
                      noise_multiplier=nm, clipping_norm=5.0, influence_fraction=fraction,
                      influence_cap=cap, seed=42)
    s.current_arrays = model(np.zeros(6))
    return s


UPDATES = np.random.default_rng(9).normal(scale=3.0, size=(4, 6))
COUNTS, IDS = [10, 40, 25, 5], [0, 1, 2, 3]


def test_noiseless_aggregate_equals_flower_global_dp():
    reference = DifferentialPrivacyServerSideFixedClipping(DeterministicFedAvg(min_available_nodes=4), 0.0, 5.0, 4)
    reference.current_arrays = model(np.zeros(6))
    expected, _ = reference.aggregate_train(1, replies(UPDATES, COUNTS, IDS))
    for fraction in (0.0, 0.5):
        got, metrics = strategy(fraction).aggregate_train(1, replies(UPDATES, COUNTS, IDS))
        np.testing.assert_allclose(flat(got), flat(expected), atol=1e-6)
        assert metrics["influence-realized-noise-sq-norm"] == 0.0


def test_reply_order_invariance_and_diagnostics():
    a, ma = strategy(0.5, nm=0.4).aggregate_train(3, replies(UPDATES, COUNTS, IDS))
    order = [2, 0, 3, 1]
    b, mb = strategy(0.5, nm=0.4).aggregate_train(
        3, replies(UPDATES[order], [COUNTS[i] for i in order], [IDS[i] for i in order]))
    np.testing.assert_array_equal(flat(a), flat(b))
    assert ma["influence-client-ids"] == IDS
    assert ma["influence-trace"] == mb["influence-trace"]
    tau = compute_stdv(0.4, 5.0, 4)
    assert ma["influence-tau"] == tau and ma["global-dp-noise-stdv"] == tau
    assert ma["influence-expected-noise-sq-norm"] == pytest.approx(tau**2 * 6)
    assert max(ma["influence-clipped-update-norms"]) <= 5.0 + 1e-5
    w = np.array(COUNTS) / sum(COUNTS)
    np.testing.assert_allclose(ma["influence-weights"], w)
    np.testing.assert_allclose(ma["influence-weighted-clipped-norms"],
                               w * np.array(ma["influence-clipped-update-norms"]))
    for key in ("influence-raw-norms", "influence-capped-norms", "influence-cap-bound",
                "influence-directional-noise-stdv"):
        assert len(ma[key]) == 4


def test_realized_noise_follows_pinned_sampler():
    noiseless, _ = strategy(0.5).aggregate_train(7, replies(UPDATES, COUNTS, IDS))
    noisy, metrics = strategy(0.5, nm=0.4).aggregate_train(7, replies(UPDATES, COUNTS, IDS))
    u = np.stack([inf.clip_update([np.asarray(r, np.float32)], 5.0)[0][0].astype(np.float64) for r in UPDATES])
    _, d = inf.leave_one_out_influence(u, inf.sample_weights(COUNTS))
    capped, _, _ = inf.cap_rows(d, 5.0)
    draw = inf.draw_noise(capped, tau=compute_stdv(0.4, 5.0, 4), fraction=0.5, rng=inf.round_generator(42, 7))
    np.testing.assert_allclose(flat(noisy) - flat(noiseless), draw.noise, atol=1e-5)
    assert metrics["influence-realized-noise-sq-norm"] == pytest.approx(float(draw.noise @ draw.noise))


def test_cap_binds_in_strategy():
    _, metrics = strategy(0.5, nm=0.1, cap=1e-3).aggregate_train(1, replies(UPDATES, COUNTS, IDS))
    assert all(metrics["influence-cap-bound"])
    np.testing.assert_allclose(metrics["influence-capped-norms"], 1e-3)


def test_identical_updates_trigger_fallback():
    same = np.tile(UPDATES[:1], (4, 1))
    _, metrics = strategy(0.5, nm=0.1).aggregate_train(1, replies(same, COUNTS, IDS))
    assert metrics["influence-fallback-isotropic"] == 1


def test_mode_is_opt_in_and_validated():
    assert "influence-noise" not in PRIVACY_MODES and "influence-noise" in EXPERIMENTAL_PRIVACY_MODES
    with pytest.raises(ValueError):
        make_strategy("fedavg", "influence-noise", num_clients=4, fraction_evaluate=1.0,
                      noise_multiplier=0.1, clipping_norm=5.0)
    with pytest.raises(ValueError):
        strategy(1.5)
