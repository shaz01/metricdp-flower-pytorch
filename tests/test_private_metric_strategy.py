"""Tests for the private-median metric-privacy strategy (WP3, D4 item 1). Synthetic, no training."""

import math
from statistics import NormalDist

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from metricdp_pytorch.metricdp_strategy import MetricPrivacyServerSideFixedClipping, pairwise_model_distances
from metricdp_pytorch.private_metric_strategy import (
    PrivateMedianMetricPrivacyServerSideFixedClipping,
    bit_noise_for_share,
    gaussian_delta,
    gaussian_epsilon,
    model_distance,
    privacy_account,
    quantile_update,
)
from metricdp_pytorch.strategy_factory import UNIT_WEIGHT_KEY


def model(*arrays):
    return ArrayRecord({f"layer-{i}": Array(np.asarray(a, dtype=float)) for i, a in enumerate(arrays)})


def replies(models):
    out = []
    for node_id, record in enumerate(models, start=1):
        request = Message(content=RecordDict(), message_type="train", dst_node_id=node_id)
        content = RecordDict({"arrays": record,
                              "metrics": MetricRecord({"num-examples": 10 * node_id, UNIT_WEIGHT_KEY: 1,
                                                       "client-id": node_id - 1})})
        out.append(Message(content=content, reply_to=request))
    return out


def strategy(num_slots=4, **kwargs):
    settings = dict(noise_multiplier=1e-9, clipping_norm=100.0, num_slots=num_slots, initial_quantile=1.0,
                    minimum_quantile=0.01, maximum_quantile=10.0, seed=0)
    settings.update(kwargs)
    return PrivateMedianMetricPrivacyServerSideFixedClipping(FedAvg(weighted_by_key=UNIT_WEIGHT_KEY), **settings)


def test_model_distance_is_the_papers_pairwise_distance():
    a = [np.array([1.0, 2.0]), np.array([[0.0, 3.0]])]
    b = [np.array([4.0, 6.0]), np.array([[1.0, 3.0]])]
    assert model_distance(a, b) == pytest.approx(pairwise_model_distances([model(*a), model(*b)])[0])
    assert model_distance(a, b) == pytest.approx((5.0 + 1.0) / 2)


def test_quantile_update_direction_and_clamp():
    assert quantile_update(1.0, 0.8, target=0.5, learning_rate=0.2, minimum=0.1, maximum=10) == pytest.approx(
        math.exp(-0.06))
    assert quantile_update(1.0, 0.2, target=0.5, learning_rate=0.2, minimum=0.1, maximum=10) > 1.0
    assert quantile_update(0.1, 1.0, target=0.5, learning_rate=5.0, minimum=0.1, maximum=10) == 0.1
    assert quantile_update(10, 0.0, target=0.5, learning_rate=5.0, minimum=0.1, maximum=10) == 10


def test_gdp_conversion_matches_closed_form():
    phi = NormalDist().cdf
    assert gaussian_delta(1.0, 0.0) == pytest.approx(phi(0.5) - phi(-0.5))
    epsilon = gaussian_epsilon(2.0, 1e-5)
    assert gaussian_delta(2.0, epsilon) == pytest.approx(1e-5, rel=1e-6)
    # Vacuous budgets (the frontier's noise, mu ~ 100) stay finite.
    large = gaussian_epsilon(135.0, 1e-5)
    assert math.isfinite(large) and gaussian_delta(135.0, large) == pytest.approx(1e-5, rel=1e-4)


def test_accounting_and_ten_percent_share():
    std = bit_noise_for_share(noise_multiplier=2.0, maximum=0.5, share=0.1)
    account = privacy_account(noise_multiplier=2.0, maximum=0.5, bit_noise_std=std, rounds=100)
    assert account.mu_update_per_round == pytest.approx(0.25)
    assert account.median_share == pytest.approx(0.1)
    assert account.mu_total == pytest.approx(10 * math.sqrt(0.25 ** 2 / 0.9))


def test_requires_equal_weights():
    with pytest.raises(ValueError):
        PrivateMedianMetricPrivacyServerSideFixedClipping(
            FedAvg(), noise_multiplier=1.0, clipping_norm=1.0, num_slots=3, initial_quantile=1.0,
            minimum_quantile=0.1, maximum_quantile=2.0)


def test_fixed_slot_denominator():
    s = strategy(num_slots=4)
    s.current_arrays = model([0.0, 0.0])
    arrays, metrics = s.aggregate_train(1, replies([model([4.0, 0.0]), model([0.0, 8.0])]))
    # Sum of updates over the 4 public slots, not the mean over the 2 participants.
    assert arrays.to_numpy_ndarrays()[0] == pytest.approx([1.0, 2.0], abs=1e-6)
    assert metrics["private-median-active-clients"] == 2
    assert metrics["private-median-true-median-distance"] == pytest.approx(6.0)


def test_noise_level_does_not_depend_on_participants():
    """IN (3 clients) and OUT (2 clients) get the same noise std; metric privacy does not."""
    clients = [model([1.0, 0.0]), model([0.0, 1.0]), model([5.0, 5.0])]
    stds, metric_stds = [], []
    for chosen in (clients, clients[:2]):
        s = strategy(num_slots=3, noise_multiplier=0.5, clipping_norm=10.0)
        s.current_arrays = model([0.0, 0.0])
        _, metrics = s.aggregate_train(1, replies(chosen))
        stds.append(metrics["private-median-noise-stdv"])
        m = MetricPrivacyServerSideFixedClipping(FedAvg(), noise_multiplier=0.5, clipping_norm=10.0,
                                                 num_sampled_clients=3)
        m.current_arrays = model([0.0, 0.0])
        _, metrics = m.aggregate_train(1, replies(chosen))
        metric_stds.append(metrics["metric-dp-noise-stdv"])
    assert stds[0] == stds[1] == pytest.approx(0.5 * 10.0 / (3 * 1.0))
    assert metric_stds[0] != pytest.approx(metric_stds[1])


def test_noise_floor_from_the_public_maximum():
    s = strategy(num_slots=4, noise_multiplier=2.0, clipping_norm=5.0, initial_quantile=10.0)
    s.current_arrays = model([0.0])
    _, metrics = s.aggregate_train(1, replies([model([0.5]), model([0.1])]))
    floor = 2.0 * 5.0 / (4 * 10.0)
    assert metrics["private-median-noise-stdv"] == pytest.approx(floor)
    # mu_update = (C / slots) / sigma_min = q_max / z.
    assert (5.0 / 4) / floor == pytest.approx(s.account(1).mu_update_per_round)


def test_quantile_tracks_the_median_without_bit_noise():
    distances = [0.5, 1.0, 2.0, 3.0, 4.0]  # median 2.0
    s = strategy(num_slots=5, bit_noise_std=1e-9, initial_quantile=0.05, learning_rate=0.3)
    for round_number in range(1, 200):
        s.current_arrays = model([0.0])
        s.aggregate_train(round_number, replies([model([d]) for d in distances]))
    assert 1.0 < s.quantile <= 3.0


def test_bits_are_the_only_participant_dependence_of_the_next_quantile():
    """Same bit-noise draw: the next quantile differs between IN and OUT only through the bit count."""
    for chosen, expected_bits in ((3, 2), (2, 1)):
        s = strategy(num_slots=3, initial_quantile=1.5, seed=7)
        s.current_arrays = model([0.0])
        noise = float(np.random.default_rng(7).normal(0.0, s.bit_noise_std))
        _, metrics = s.aggregate_train(1, replies([model([d]) for d in (1.0, 2.0, 0.5)][:chosen][::-1]))
        assert metrics["private-median-noisy-fraction"] == pytest.approx((expected_bits + noise) / 3)


def test_a_round_whose_updates_all_collapse_keeps_the_model():
    """Every client returns the current model (zero-norm updates): Flower's clipping would divide by zero."""
    s = strategy(num_slots=3)
    current = model([1.0, 2.0])
    s.current_arrays = current
    before = s.quantile
    arrays, metrics = s.aggregate_train(1, replies([model([1.0, 2.0])] * 3))
    assert arrays is None
    assert metrics["private-median-aggregation-collapsed"] == 1.0
    assert s.rounds_released == 1 and s.quantile != before  # the quantile release still happened and is counted
