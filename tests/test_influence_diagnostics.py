"""Per-client influence diagnostics: values, IDs, and no effect on aggregation."""
import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict

from metricdp_pytorch.influence_diagnostics import (
    InfluenceLoggingStrategy,
    influence_metrics,
    per_client_influence,
)
from metricdp_pytorch.strategy_factory import make_strategy


def test_values_match_hand_computation():
    updates = [np.array([3.0, 4.0]), np.array([0.0, 1.0])]  # norms 5 and 1
    rows = per_client_influence(updates, [7, 2], [1, 3], clipping_norm=2.5)
    clipped = [np.array([1.5, 2.0]), np.array([0.0, 1.0])]
    g = 0.25 * clipped[0] + 0.75 * clipped[1]
    assert rows[7]["update_norm_before_clipping"] == pytest.approx(5.0)
    assert rows[7]["clipped"] is True and rows[2]["clipped"] is False
    assert rows[7]["aggregation_weight"] == 0.25 and rows[2]["num_examples"] == 3
    assert rows[7]["distance_from_weighted_clipped_average"] == pytest.approx(np.linalg.norm(clipped[0] - g))
    cosine = clipped[0] @ g / (np.linalg.norm(clipped[0]) * np.linalg.norm(g))
    assert rows[7]["cosine_with_weighted_clipped_average"] == pytest.approx(cosine)
    assert rows[7]["leave_one_out_influence_norm"] == pytest.approx(0.25 / 0.75 * np.linalg.norm(clipped[0] - g))
    # LOO identity: g without client 7 is client 2's update, and g - g_{-7} has that norm.
    assert rows[7]["leave_one_out_influence_norm"] == pytest.approx(np.linalg.norm(g - clipped[1]))


def test_vanilla_never_clips_and_inputs_untouched():
    updates = [np.array([30.0, 0.0]), np.array([0.0, 1.0])]
    snapshot = [u.copy() for u in updates]
    rows = per_client_influence(updates, [0, 1], [1, 1], clipping_norm=None)
    assert not rows[0]["clipped"]
    assert rows[0]["distance_from_weighted_clipped_average"] == pytest.approx(np.linalg.norm([15.0, -0.5]))
    assert all(np.array_equal(a, b) for a, b in zip(updates, snapshot))
    flat = influence_metrics(rows)
    assert flat["influence-client-ids"] == [0, 1] and flat["influence-clipped"] == [0, 0]
    with pytest.raises(ValueError):
        per_client_influence(updates, [0, 0], [1, 1], None)


def _model(values):
    return ArrayRecord({"w": Array(np.asarray(values, dtype=np.float32))})


def _replies():
    replies = []
    for client, values, count in ((0, [9.0, 0.0, 1.0], 10), (1, [0.5, 0.5, 0.5], 30), (2, [-1.0, 2.0, 0.0], 20)):
        request = Message(content=RecordDict(), message_type="train", dst_node_id=client + 1)
        content = RecordDict({"arrays": _model(values),
                              "metrics": MetricRecord({"client-id": client, "num-examples": count})})
        replies.append(Message(content=content, reply_to=request))
    return replies


def _aggregate(privacy, logged):
    strategy = make_strategy("fedavg", privacy, num_clients=3, fraction_evaluate=1.0,
                             noise_multiplier=0.3, clipping_norm=2.0)
    current = _model([0.0, 0.0, 0.0])
    if privacy != "vanilla":
        strategy.current_arrays = current
    if logged:
        strategy = InfluenceLoggingStrategy(strategy, clipping_norm=None if privacy == "vanilla" else 2.0,
                                            id_map=(4, 7, 9))
        strategy.current_arrays = current
    np.random.seed(0)  # Flower's DP noise draws from numpy's global RNG
    arrays, metrics = strategy.aggregate_train(1, _replies())
    return arrays.to_numpy_ndarrays()[0], metrics, np.random.random()


@pytest.mark.parametrize("privacy", ["vanilla", "global-dp", "metric-privacy"])
def test_logging_leaves_aggregation_and_rng_bit_identical(privacy):
    plain, plain_metrics, plain_rng = _aggregate(privacy, logged=False)
    logged, metrics, logged_rng = _aggregate(privacy, logged=True)
    assert plain.tobytes() == logged.tobytes()
    assert plain_rng == logged_rng  # no extra draws from the global RNG
    assert not any(key.startswith("influence-") for key in plain_metrics)
    assert {k: v for k, v in metrics.items() if not k.startswith("influence-")} == dict(plain_metrics)
    assert metrics["influence-client-ids"] == [4, 7, 9]  # canonical IDs through the view map
    assert metrics["influence-clipped"] == ([0, 0, 0] if privacy == "vanilla" else [1, 0, 1])
    assert metrics["influence-aggregation-weight"] == pytest.approx([1 / 6, 1 / 2, 1 / 3])


def _equal_replies():
    replies = _replies()
    for reply in replies:
        next(iter(reply.content.metric_records.values()))["unit-weight"] = 1
    return replies


@pytest.mark.parametrize("privacy", ["vanilla", "global-dp", "metric-privacy"])
def test_equal_weighting_logs_one_over_n_and_equal_weighted_loo(privacy):
    clip = None if privacy == "vanilla" else 2.0
    strategy = make_strategy("fedavg", privacy, num_clients=3, fraction_evaluate=1.0,
                             noise_multiplier=0.0, clipping_norm=2.0, weighting="equal")
    current = _model([0.0, 0.0, 0.0])
    if privacy != "vanilla":
        strategy.current_arrays = current
    strategy = InfluenceLoggingStrategy(strategy, clipping_norm=clip)
    assert strategy.weight_key == "unit-weight"
    strategy.current_arrays = current
    arrays, metrics = strategy.aggregate_train(1, _equal_replies())
    assert metrics["influence-aggregation-weight"] == pytest.approx([1 / 3] * 3)
    assert metrics["influence-num-examples"] == [10, 30, 20]
    updates = [np.array(v) for v in ([9.0, 0.0, 1.0], [0.5, 0.5, 0.5], [-1.0, 2.0, 0.0])]
    clipped = [u if clip is None else u * min(1.0, clip / np.linalg.norm(u)) for u in updates]
    g = np.mean(clipped, axis=0)
    np.testing.assert_allclose(arrays.to_numpy_ndarrays()[0], g, rtol=1e-6, atol=1e-6)  # noise 0
    loo = [0.5 * np.linalg.norm(c - g) for c in clipped]  # w/(1-w) = (1/3)/(2/3)
    assert metrics["influence-leave-one-out-influence-norm"] == pytest.approx(loo)
    for i, c in enumerate(clipped):  # LOO identity: g - mean of the others
        others = np.mean([d for j, d in enumerate(clipped) if j != i], axis=0)
        assert loo[i] == pytest.approx(np.linalg.norm(g - others))


@pytest.mark.parametrize("privacy", ["vanilla", "global-dp", "metric-privacy"])
def test_pairwise_matrix_logged_for_every_mode_on_raw_models(privacy):
    from metricdp_pytorch.metricdp_strategy import pairwise_model_distances

    plain, plain_metrics, plain_rng = _aggregate(privacy, logged=False)
    logged, metrics, logged_rng = _aggregate(privacy, logged=True)
    assert plain.tobytes() == logged.tobytes() and plain_rng == logged_rng
    raw = [r.content["arrays"] for r in _replies()]  # unclipped client models, ids 0,1,2 -> 4,7,9
    assert metrics["influence-pairwise-distances"] == pairwise_model_distances(raw)
    assert metrics["influence-pairwise-client-i"] == [4, 4, 7]
    assert metrics["influence-pairwise-client-j"] == [7, 9, 9]
    if privacy == "metric-privacy":  # identical to the mechanism's own d measurement
        assert metrics["influence-pairwise-distances"] == pytest.approx(metrics["metric-dp-pairwise-distances"])
