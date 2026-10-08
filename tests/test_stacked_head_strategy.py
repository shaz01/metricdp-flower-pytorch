"""Equivalence and contract tests for the Flower stacked head-step strategy."""

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, MetricRecord, RecordDict

from metricdp_pytorch.stacked_head_strategy import (
    CONTRAST,
    HEAD_BIAS_KEY,
    HEAD_WEIGHT_KEY,
    StackedHeadConfig,
    StackedHeadStrategy,
    class_probabilities,
    client_contribution,
    client_update,
    head_theta,
    projection_basis,
    public_class_gradients,
    with_head,
)

FEATURES = 16
D = FEATURES + 1


def synthetic(seed=3, n=64):
    rng = np.random.default_rng(seed)
    x = np.column_stack((rng.normal(size=(n, FEATURES)), np.ones(n)))
    y = np.tile(np.arange(4), n // 4)
    theta = rng.normal(size=(3, D)) * 0.1
    return rng, x, y, theta


def make_arrays(theta, extra=None):
    weights = CONTRAST @ theta
    arrays = {HEAD_WEIGHT_KEY: Array(weights[:, :-1].astype(np.float32)), HEAD_BIAS_KEY: Array(weights[:, -1].astype(np.float32))}
    if extra is not None:
        arrays["body.weight"] = Array(extra)
    return ArrayRecord(arrays)


def reply(client_id, vector):
    content = RecordDict({"update": ArrayRecord({"vector": Array(vector)}), "metrics": MetricRecord({"client-id": client_id})})
    return Message(content=content, dst_node_id=0, message_type="train")


def test_contribution_matches_research_probe_and_respects_cap():
    from research.calculations.public_residual_probe import bounded_public
    from research.calculations.class_conditional_probe import client_queries
    rng, x, y, theta = synthetic()
    px, py = x[:32], y[:32]
    gamma = public_class_gradients(px, py, theta)
    d = 12
    basis = projection_basis(px, theta, d)
    cohort_x = np.column_stack((rng.normal(size=(8 * 40, FEATURES)), np.ones(8 * 40)))
    cohort_y = rng.integers(0, 4, size=8 * 40)
    parts = [np.arange(i * 40, (i + 1) * 40) for i in range(8)]
    config = StackedHeadConfig(cap=0.05, eta=7.0, dimension=d, risk=0.65)
    queries, _ = client_queries(cohort_x, cohort_y, parts, theta, gamma, np.ones(4) / 4)
    expected, _ = bounded_public(queries["target_center"].reshape(8, -1) @ basis, config.cap)
    for i, idx in enumerate(parts):
        got = client_contribution(cohort_x[idx], cohort_y[idx], theta, gamma, basis, config)
        assert np.allclose(got, expected[i], atol=1e-12)
        assert np.linalg.norm(got) <= config.cap / 8 + 1e-12


def test_basis_matches_research_geometry():
    from research.calculations.public_residual_probe import public_geometry, public_hessian
    rng, x, y, theta = synthetic()
    assert FEATURES + 1 == 17
    h = public_hessian(x, theta)
    for d in (1, 3, 12, 51):
        assert np.allclose(projection_basis(x, theta, d), public_geometry(h, d, "euclidean")["basis"], atol=1e-10)


def test_noise_shares_have_calibrated_variance():
    config = StackedHeadConfig(cap=0.01, eta=1.0, dimension=6, risk=0.8)
    rng = np.random.default_rng(1)
    total = sum(np.stack([client_update(np.zeros((0, D)), np.zeros(0, dtype=int), np.zeros((3, D)), np.zeros((4, 3, D)), np.eye(3 * D)[:, :6], config, rng=rng) for _ in range(20000)]) for _ in range(8))
    assert abs(total.var(axis=0).mean() / (config.sigma**2 * 8 / 7) - 1) < 0.05


def test_strategy_applies_validation_gated_step_and_preserves_other_arrays():
    from research.calculations.client_energy_filter_probe import scores
    rng, x, y, theta = synthetic()
    gamma = public_class_gradients(x[:32], y[:32], theta)
    d = 12
    basis = projection_basis(x[:32], theta, d)
    config = StackedHeadConfig(cap=0.05, eta=40.0, dimension=d, risk=0.65, noise_free=True)
    cohort_x = np.column_stack((rng.normal(size=(8 * 40, FEATURES)), np.ones(8 * 40)))
    cohort_y = rng.integers(0, 4, size=8 * 40)
    parts = [np.arange(i * 40, (i + 1) * 40) for i in range(8)]
    vx, vy = x, y
    strategy = StackedHeadStrategy(config, basis, gamma, lambda thetas: scores(vx, vy, thetas)[0])
    body = np.arange(6, dtype=np.float32)
    arrays = make_arrays(theta, extra=body)
    strategy.current_arrays = arrays
    replies = [reply(i, client_update(cohort_x[p], cohort_y[p], theta, gamma, basis, config, noise=np.zeros(d))) for i, p in enumerate(parts)][::-1]
    out, metrics = strategy.aggregate_train(1, replies)
    total = sum(client_contribution(cohort_x[p], cohort_y[p], theta, gamma, basis, config) for p in parts)
    candidates = np.stack([theta - config.eta * m * (total @ basis.T).reshape(3, D) for m in config.multipliers])
    ce = scores(vx, vy, candidates)[0]
    chosen = candidates[int(ce.argmin())]
    assert metrics["multiplier"] == pytest.approx(config.multipliers[int(ce.argmin())])
    assert np.allclose(head_theta(out), chosen, atol=1e-5)
    assert np.array_equal(out["body.weight"].numpy(), body)
    assert np.allclose(class_probabilities(x, head_theta(out)), class_probabilities(x, chosen), atol=1e-5)


def test_zero_multiplier_returns_base_when_validation_prefers_it():
    rng, x, y, theta = synthetic()
    gamma = public_class_gradients(x[:32], y[:32], theta)
    basis = projection_basis(x[:32], theta, 3)
    config = StackedHeadConfig(cap=0.05, eta=5.0, dimension=3, risk=0.65, noise_free=True)
    strategy = StackedHeadStrategy(config, basis, gamma, lambda thetas: np.arange(len(thetas), dtype=float))
    strategy.current_arrays = make_arrays(theta)
    out, metrics = strategy.aggregate_train(1, [reply(0, np.ones(3))])
    assert metrics["multiplier"] == 0.0 and np.allclose(head_theta(out), theta, atol=1e-5)


def test_failed_replies_are_skipped_and_empty_rounds_return_none():
    rng, x, y, theta = synthetic()
    gamma = public_class_gradients(x[:32], y[:32], theta)
    basis = projection_basis(x[:32], theta, 3)
    strategy = StackedHeadStrategy(StackedHeadConfig(cap=0.05, eta=5.0, dimension=3, risk=0.65, noise_free=True), basis, gamma, lambda t: np.zeros(len(t)))
    strategy.current_arrays = make_arrays(theta)
    assert strategy.aggregate_train(1, []) == (None, None)


class StubGrid:
    def __init__(self, count):
        self.ids = list(range(1, count + 1))

    def get_node_ids(self):
        return self.ids


def test_configure_train_ships_model_public_geometry_and_construction_config():
    rng, x, y, theta = synthetic()
    gamma = public_class_gradients(x[:32], y[:32], theta)
    basis = projection_basis(x[:32], theta, 5)
    config = StackedHeadConfig(cap=0.02, eta=9.0, dimension=5, risk=0.8)
    strategy = StackedHeadStrategy(config, basis, gamma, lambda t: np.zeros(len(t)), min_train_nodes=8, min_available_nodes=8)
    arrays = make_arrays(theta)
    messages = list(strategy.configure_train(2, arrays, ConfigRecord(), StubGrid(8)))
    assert len(messages) == 8 and sorted(m.metadata.dst_node_id for m in messages) == list(range(1, 9))
    content = messages[0].content
    assert np.allclose(content["public"]["basis"].numpy(), basis) and np.allclose(content["public"]["class-gradients"].numpy(), gamma)
    sent = content["config"]
    assert (sent["server-round"], sent["dimension"], sent["cap"], sent["risk"], sent["mode"], sent["peers"]) == (2, 5, 0.02, 0.8, "target_center", 7)
    assert strategy.current_arrays is arrays
