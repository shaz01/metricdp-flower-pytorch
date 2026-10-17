"""Synthetic tests for the frozen-head federation (no dataset, no extractor, CPU only)."""

from __future__ import annotations

import numpy as np
import pytest

from experiments.frozen_head.data import build_layout
from experiments.frozen_head.fedavg import Head, HeadConfig, clip_update, run
from experiments.frozen_head.public_only import Standardizer, fit_public_only
from metricdp_pytorch.utils.split_data import split_stratified
from results.cia_frontier.eurosat_frontier.data import dirichlet_partitions

CLASSES, DIM = 4, 6


def _blobs(n: int, seed: int, spread: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, CLASSES, n)
    centres = np.random.default_rng(99).normal(0, 3, (CLASSES, DIM))
    return centres[labels] + rng.normal(0, spread, (n, DIM)), labels


def _config(**overrides) -> HeadConfig:
    base = dict(rounds=3, local_epochs=2, local_lr=0.1, batch_size=8, clip=1.0, server_lr=1.0,
                noise_multiplier=0.0, num_slots=4)
    return HeadConfig(**{**base, **overrides})


def _clients(count: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    return [_blobs(30, seed) for seed in range(count)]


def test_gradient_matches_finite_differences():
    head = Head(CLASSES, DIM)
    features, labels = _blobs(20, 1)
    theta = np.random.default_rng(2).normal(0, 0.3, head.size)
    _, grad = head.loss_and_grad(theta, features, labels)
    step = 1e-6
    numeric = np.array([(head.loss_and_grad(theta + step * e, features, labels)[0]
                         - head.loss_and_grad(theta - step * e, features, labels)[0]) / (2 * step)
                        for e in np.eye(head.size)])
    assert np.allclose(grad, numeric, atol=1e-7)


def test_clip_update():
    assert np.allclose(clip_update(np.array([3.0, 4.0]), 1.0), [0.6, 0.8])
    assert np.allclose(clip_update(np.array([0.3, 0.4]), 1.0), [0.3, 0.4])


def test_noise_std_and_independence_between_noise_seeds():
    head = Head(50, 40)  # 2,050 coordinates
    clients = [(np.zeros((2, 40)), np.zeros(2, dtype=int))] * 4
    config = _config(noise_multiplier=2.0, rounds=1, local_lr=0.0)  # zero updates: release = noise
    first = run(head, np.zeros(head.size), clients, config, client_seed=0, noise_seed=10)
    same_noise = run(head, np.zeros(head.size), clients, config, client_seed=5, noise_seed=10)
    other_noise = run(head, np.zeros(head.size), clients, config, client_seed=0, noise_seed=11)
    assert first["noise_std"] == pytest.approx(2.0 * 1.0 / 4)
    assert np.std(first["theta"]) == pytest.approx(0.5, rel=0.05)
    assert np.array_equal(first["theta"], same_noise["theta"])
    assert abs(np.corrcoef(first["theta"], other_noise["theta"])[0, 1]) < 0.1


def test_denominator_is_the_public_slot_count():
    head = Head(CLASSES, DIM)
    clients = [_blobs(30, 7)] * 4  # identical clients, identical updates
    config = _config(rounds=1, clip=100.0)
    full = run(head, np.zeros(head.size), clients, config, client_seed=0, noise_seed=0)
    half = run(head, np.zeros(head.size), clients, config, client_seed=0, noise_seed=0, active=[0, 1])
    single = run(head, np.zeros(head.size), clients, _config(rounds=1, clip=100.0, num_slots=1),
                 client_seed=0, noise_seed=0, active=[0])
    # Client ids draw different batch orders, so the updates agree only approximately.
    assert np.linalg.norm(half["theta"]) == pytest.approx(np.linalg.norm(full["theta"]) / 2, rel=0.05)
    assert np.linalg.norm(single["theta"]) == pytest.approx(np.linalg.norm(full["theta"]), rel=0.05)
    with pytest.raises(ValueError):
        run(head, np.zeros(head.size), clients * 2, config, client_seed=0, noise_seed=0)


def test_noise_free_training_learns_and_is_deterministic():
    head = Head(CLASSES, DIM)
    clients = _clients()
    test = _blobs(200, 50)
    config = _config(rounds=10, clip=10.0)
    first = run(head, np.zeros(head.size), clients, config, client_seed=3, noise_seed=0, evaluate={"test": test})
    second = run(head, np.zeros(head.size), clients, config, client_seed=3, noise_seed=0)
    assert first["accuracy"]["test"][-1] > 0.9
    assert np.array_equal(first["theta"], second["theta"])


def test_keep_updates_shapes_and_clipping():
    head = Head(CLASSES, DIM)
    out = run(head, np.zeros(head.size), _clients(), _config(clip=0.05), client_seed=0, noise_seed=0,
              keep_updates=True)
    assert out["updates"].shape == (3, 4, head.size)
    assert out["round_start_theta"].shape == (3, head.size)
    assert np.all(np.linalg.norm(out["updates"], axis=2) <= 0.05 + 1e-12)
    assert np.allclose(out["round_start_theta"][1], out["updates"][0].sum(axis=0) / 4)


def test_public_only_head_matches_its_sklearn_model():
    head = Head(CLASSES, DIM)
    features, labels = _blobs(200, 3, spread=2.0)
    scale = Standardizer.fit(features)
    fit = fit_public_only(head, scale(features), labels)
    test_features, test_labels = _blobs(400, 4, spread=2.0)
    assert head.accuracy(fit.theta, scale(test_features), test_labels) > 0.8
    assert set(fit.cv_accuracy) and fit.inverse_l2 in fit.cv_accuracy


def _synthetic_labels(n: int, seed: int) -> tuple[int, ...]:
    return tuple(np.random.default_rng(seed).integers(0, 10, n).tolist())


def test_layout_matches_the_frontier_partitioning():
    train, test = _synthetic_labels(4000, 0), _synthetic_labels(400, 1)
    layout = build_layout(train, test, num_clients=8, layout_seed=42)
    parts = dirichlet_partitions(train, 8, 0.3, 42)
    for t, part in enumerate(parts):
        expected_train, expected_test = split_stratified(train, part, 0.8, seed=42 + t)
        assert np.array_equal(layout.client_train[t], sorted(expected_train))
        assert np.array_equal(layout.client_test[t], sorted(expected_test))
    public, server_test = split_stratified(test, list(range(400)), 0.5, seed=42)
    assert np.array_equal(layout.public, sorted(public)) and np.array_equal(layout.server_test, sorted(server_test))


def test_split_layout_partitions_each_base_client():
    train, test = _synthetic_labels(4000, 0), _synthetic_labels(400, 1)
    base = dirichlet_partitions(train, 8, 0.3, 42)
    split = build_layout(train, test, num_clients=40, layout_seed=42, split_factor=5)
    for t, part in enumerate(base):
        members = np.concatenate([np.concatenate([split.client_train[t * 5 + k], split.client_test[t * 5 + k]])
                                  for k in range(5)])
        assert sorted(members.tolist()) == sorted(part)
    assert split.client_sizes.sum() == 4000
    assert len(split.top_decile()) == 4
    with pytest.raises(ValueError):
        build_layout(train, test, num_clients=41, layout_seed=42, split_factor=5)


def test_crowd_ratios_match_dense_formulas():
    from experiments.frozen_head.crowd import compose, full_release_ratio, statistic_ratio

    rng = np.random.default_rng(0)
    n, d, sigma, clip = 9, 7, 0.3, 1.0
    updates = rng.normal(0, 0.4, (n, d))
    others = np.delete(updates, 2, axis=0)
    sigma_b = (n - 1) * np.cov(others, rowvar=False)
    dense = sigma ** 2 * np.eye(d) + sigma_b / n ** 2
    u = updates[2]
    expected = sigma * np.sqrt(u @ np.linalg.solve(dense, u)) / clip
    assert full_release_ratio(updates, 2, sigma, clip, n) == pytest.approx(expected, rel=1e-10)
    w = rng.normal(size=d)
    rho = sigma * np.linalg.norm(w) / np.sqrt(sigma ** 2 * w @ w + w @ sigma_b @ w / n ** 2)
    assert statistic_ratio(updates, 2, w, sigma, n) == pytest.approx(rho, rel=1e-10)
    # Large noise: the crowd no longer helps (ratio -> ||u|| / C, rho -> 1).
    assert full_release_ratio(updates, 2, 1e3, clip, n) == pytest.approx(np.linalg.norm(u) / clip, rel=1e-6)
    assert statistic_ratio(updates, 2, w, 1e3, n) == pytest.approx(1.0, abs=1e-6)
    assert compose(np.array([0.6, 0.8])) == pytest.approx(np.sqrt(0.5))
