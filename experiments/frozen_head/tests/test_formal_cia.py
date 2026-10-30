"""P5 mechanism options and analysis on synthetic data (no training on EuroSAT, no dataset)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.frozen_head import formal_cia
from experiments.frozen_head.fedavg import Head, HeadConfig, clip_update, local_update, max_pairwise_layer_distance, run


def _federation(clients=6, records=20, dim=4, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    head = Head(classes, dim)
    data = [(rng.normal(size=(records, dim)), rng.integers(classes, size=records)) for _ in range(clients)]
    return head, data, np.zeros(head.size)


def _config(z=1.0, slots=6, rounds=3):
    return HeadConfig(rounds=rounds, local_epochs=1, local_lr=0.1, batch_size=8, clip=0.5, server_lr=1.0,
                      noise_multiplier=z, num_slots=slots)


def _reference_run(head, init, data, config, client_seed, noise_seed):
    """The pre-P5 loop, written out: the default path must stay bit-identical to it."""
    noise_rng = np.random.default_rng(noise_seed)
    std = config.noise_multiplier * config.clip / config.num_slots
    theta = init.copy()
    for t in range(config.rounds):
        total = np.zeros(head.size)
        for client, (x, y) in enumerate(data):
            update = local_update(head, theta, x, y, config, np.random.default_rng([client_seed, client, t]))
            total += clip_update(update, config.clip)
        theta = theta + config.server_lr * (total / config.num_slots + noise_rng.normal(0.0, std, head.size))
    return theta


def test_default_path_bit_identical():
    head, data, init = _federation()
    config = _config()
    out = run(head, init, data, config, client_seed=3, noise_seed=4)
    assert np.array_equal(out["theta"], _reference_run(head, init, data, config, 3, 4))
    assert "noise_std_by_round" not in out and "sampled_by_round" not in out


def test_max_pairwise_layer_distance_matches_strategy_definition():
    head = Head(3, 4)
    models = np.random.default_rng(1).normal(size=(5, head.size))
    expected = max(np.mean([np.linalg.norm(models[i, :12] - models[j, :12]), np.linalg.norm(models[i, 12:] - models[j, 12:])])
                   for i in range(5) for j in range(i + 1, 5))
    assert max_pairwise_layer_distance(head, models) == pytest.approx(expected, rel=1e-10)


def test_metric_calibration_divides_noise_by_distance():
    head, data, init = _federation()
    config = _config(z=2.0)
    out = run(head, init, data, config, client_seed=3, noise_seed=4, calibration="metric")
    base = config.noise_multiplier * config.clip / config.num_slots
    assert len(out["distance_by_round"]) == config.rounds
    for d, std in zip(out["distance_by_round"], out["noise_std_by_round"]):
        assert d > 0 and std == pytest.approx(base / d)


def test_metric_calibration_rejects_sampling():
    head, data, init = _federation()
    with pytest.raises(ValueError):
        run(head, init, data, _config(), client_seed=0, noise_seed=0, calibration="metric", sampling_rate=0.5,
            sampling_seed=1)


def test_sampling_denominator_noise_and_rate():
    head, data, init = _federation(clients=40)
    config = _config(z=0.0, slots=40, rounds=1)
    out = run(head, init, data, config, client_seed=1, noise_seed=2, sampling_rate=0.25, sampling_seed=5)
    sampled = out["sampled_by_round"][0]
    total = sum(clip_update(local_update(head, init, *data[c], config, np.random.default_rng([1, c, 0])), config.clip)
                for c in sampled)
    assert np.allclose(out["theta"], init + total / (0.25 * 40))
    noisy = run(head, init, data, _config(z=1.5, slots=40, rounds=1), client_seed=1, noise_seed=2, sampling_rate=0.25,
                sampling_seed=5)
    assert noisy["noise_std_by_round"][0] == pytest.approx(1.5 * 0.5 / 10)
    many = run(head, init, data, _config(z=0.0, slots=40, rounds=40), client_seed=1, noise_seed=2, sampling_rate=0.25,
               sampling_seed=6)
    assert abs(np.mean([len(s) for s in many["sampled_by_round"]]) / 40 - 0.25) < 0.03


def test_sampling_needs_its_own_seed_and_is_private_to_it():
    head, data, init = _federation(clients=20)
    config = _config(slots=20)
    with pytest.raises(ValueError):
        run(head, init, data, config, client_seed=1, noise_seed=2, sampling_rate=0.25)
    a = run(head, init, data, config, client_seed=1, noise_seed=2, sampling_rate=0.25, sampling_seed=7)
    b = run(head, init, data, config, client_seed=1, noise_seed=2, sampling_rate=0.25, sampling_seed=8)
    assert a["sampled_by_round"] != b["sampled_by_round"]


def test_design_and_seeds():
    design = formal_cia.design()
    assert design.shape == (32, 32) and np.all(design.sum(axis=0) == 16) and np.all(design.sum(axis=1) == 16)
    seeds = [formal_cia.replicate_seeds(j) for j in range(32)]
    assert len({s for triple in seeds for s in triple}) == 96
    assert formal_cia.replicate_seeds(5) == seeds[5]


def test_job_grid():
    jobs = formal_cia.jobs()
    assert len(jobs) == 2 * 13 * 32
    assert len({j.name for j in jobs}) == len(jobs)


def test_schedule_statistic_window_one():
    rng = np.random.default_rng(0)
    losses = rng.normal(size=(3, 4, 6))
    schedules = rng.random((3, 4, 5)) < 0.25
    z = formal_cia.schedule_statistic(losses, schedules, 0.25)
    drops = losses[..., :-1] - losses[..., 1:]
    for c in range(3):
        for j in range(4):
            expected = sum((schedules[c, j, t] - 0.25) * (drops[c, j, t] - drops[c, :, t].mean()) for t in range(5))
            assert z[c, j] == pytest.approx(expected)


def _point(shift, accuracy, seed):
    rng = np.random.default_rng(seed)
    member = formal_cia.design()
    scores = rng.normal(size=member.shape) + shift * member
    return formal_cia.Point(scores, member, np.full(32, accuracy))


def test_matched_labels_and_testability():
    mech = [_point(0.0, 0.9, 1), _point(0.0, 0.7, 2)]
    ref = [_point(1.5, 0.9, 3), _point(1.5, 0.7, 4)]
    result = formal_cia.matched(mech, ref, 0.8, resamples=50)
    assert result["testability"] == "testable" and result["beats"] and result["a1_outcome"] == "better"
    assert formal_cia.matched(mech, ref, 0.95, resamples=10)["testability"] == "not testable"
    assert "beats" not in formal_cia.matched(mech, ref, 0.6, resamples=10)


def test_target_at_curve_end_is_not_testable_by_the_gate_interval():
    mech = [_point(0.0, 0.9, 1), _point(0.0, 0.7, 2)]
    # mean exactly 0.70 at the curve's lower end; leaving out pair (0, 1) lifts it above the target
    mech[1].accuracy[:] = np.r_[np.full(2, 0.60), np.full(30, (22.4 - 1.2) / 30)]
    ref = [_point(0.5, 0.9, 3), _point(0.5, 0.6, 4)]
    result = formal_cia.matched(mech, ref, mech[1].accuracy.mean(), resamples=10)
    assert result["testability"] == "not testable by the gate interval"
    assert not math.isnan(result["gate"]["estimate"]) and "beats" not in result
