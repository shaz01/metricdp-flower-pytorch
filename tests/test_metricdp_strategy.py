"""Tests for metric-aware DP distance calibration."""

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from metricdp_pytorch.metricdp_strategy import (
    MAX_CLIENTS_FOR_PAIRWISE_LOGGING,
    MetricPrivacyServerSideFixedClipping,
    maximum_pairwise_model_distance,
    pairwise_model_distances,
)


def model(*arrays: np.ndarray) -> ArrayRecord:
    """Build a named ArrayRecord for a synthetic model."""
    return ArrayRecord(
        {f"layer-{index}": Array(array) for index, array in enumerate(arrays)}
    )


def test_maximum_pairwise_model_distance() -> None:
    """Use the maximum of the mean layer-wise Euclidean distances."""
    models = [
        model(np.array([0.0, 0.0]), np.array([0.0])),
        model(np.array([3.0, 4.0]), np.array([1.0])),
        model(np.array([0.0, 0.0]), np.array([4.0])),
    ]

    # Pair distances are 3.0, 2.0, and 4.0 respectively.
    assert maximum_pairwise_model_distance(models) == pytest.approx(4.0)


def test_pairwise_model_distances_returns_full_distribution() -> None:
    """Return every pairwise distance, not just the max."""
    models = [
        model(np.array([0.0, 0.0]), np.array([0.0])),
        model(np.array([3.0, 4.0]), np.array([1.0])),
        model(np.array([0.0, 0.0]), np.array([4.0])),
    ]

    distances = pairwise_model_distances(models)

    assert sorted(distances) == pytest.approx([2.0, 3.0, 4.0])
    assert max(distances) == maximum_pairwise_model_distance(models)


def test_distance_requires_two_models() -> None:
    """Reject a round in which pairwise distance cannot be computed."""
    with pytest.raises(ValueError, match="at least two"):
        maximum_pairwise_model_distance([model(np.array([0.0]))])


def test_distance_rejects_incompatible_models() -> None:
    """Reject client models with different array shapes."""
    with pytest.raises(ValueError, match="matching array shapes"):
        maximum_pairwise_model_distance([model(np.zeros(2)), model(np.zeros(3))])


def test_modern_strategy_aggregates_message_replies() -> None:
    """Integrate with Flower's message-based FedAvg strategy."""
    strategy = MetricPrivacyServerSideFixedClipping(
        strategy=FedAvg(),
        noise_multiplier=0.0,
        clipping_norm=10.0,
        num_sampled_clients=2,
    )
    strategy.current_arrays = model(np.array([0.0, 0.0]))

    replies = []
    for node_id, values in enumerate(([1.0, 0.0], [3.0, 4.0]), start=1):
        request = Message(
            content=RecordDict(), message_type="train", dst_node_id=node_id
        )
        content = RecordDict(
            {
                "arrays": model(np.asarray(values)),
                "metrics": MetricRecord({"num-examples": 1}),
            }
        )
        replies.append(Message(content=content, reply_to=request))

    arrays, metrics = strategy.aggregate_train(1, replies)

    assert arrays is not None
    assert metrics is not None
    assert arrays.to_numpy_ndarrays()[0] == pytest.approx(np.array([2.0, 2.0]))
    assert metrics["metric-dp-distance"] == pytest.approx(np.sqrt(20.0))
    assert metrics["metric-dp-pairwise-distances"] == pytest.approx(
        [np.sqrt(20.0)]
    )
    assert metrics["metric-dp-pairwise-client-i"] == [0]
    assert metrics["metric-dp-pairwise-client-j"] == [1]
    assert metrics["metric-dp-distance-min"] == pytest.approx(np.sqrt(20.0))
    # Only one pair of client models, so mean/median collapse to the max.
    assert metrics["metric-dp-distance-mean"] == pytest.approx(np.sqrt(20.0))
    assert metrics["metric-dp-distance-median"] == pytest.approx(np.sqrt(20.0))
    assert metrics["metric-dp-distance-count"] == 1
    assert metrics["metric-dp-distance-invalid"] == 0.0
    assert metrics["metric-dp-noise-stdv"] == 0.0
    assert metrics["dp-update-norms-before-clipping"] == pytest.approx([1.0, 5.0])
    assert metrics["dp-client-clipped"] == [0, 0]
    assert metrics["dp-fraction-clipped"] == 0.0
    assert metrics["dp-parameter-count"] == 2
    assert metrics["dp-expected-noise-l2-norm"] == 0.0


def test_pairwise_lists_omitted_above_client_count_threshold() -> None:
    """Full pairwise-distance/client-id lists must be dropped past the
    threshold to bound run-JSON size, while every summary statistic that
    the calibration itself relies on stays present and correct.

    Grows as C(num_clients, 2); at high client counts (measured at 256
    clients / 120 rounds during the CIFAR-100 client-scaling experiment) the
    raw lists alone balloon a single run's JSON past GitHub's 100MB push
    limit. Only the diagnostic verbosity is gated here -- the calibration
    math (``raw_distance = max(distances)``) is unaffected either way.
    """
    num_clients = MAX_CLIENTS_FOR_PAIRWISE_LOGGING + 1
    strategy = MetricPrivacyServerSideFixedClipping(
        strategy=FedAvg(),
        noise_multiplier=0.0,
        clipping_norm=10.0,
        num_sampled_clients=num_clients,
    )
    strategy.current_arrays = model(np.zeros(num_clients))

    replies = []
    for node_id in range(1, num_clients + 1):
        values = np.zeros(num_clients)
        values[node_id - 1] = float(node_id)
        request = Message(
            content=RecordDict(), message_type="train", dst_node_id=node_id
        )
        content = RecordDict(
            {
                "arrays": model(values),
                "metrics": MetricRecord({"num-examples": 1}),
            }
        )
        replies.append(Message(content=content, reply_to=request))

    _arrays, metrics = strategy.aggregate_train(1, replies)

    assert metrics is not None
    assert "metric-dp-pairwise-distances" not in metrics
    assert "metric-dp-pairwise-client-i" not in metrics
    assert "metric-dp-pairwise-client-j" not in metrics
    expected_pair_count = num_clients * (num_clients - 1) // 2
    assert metrics["metric-dp-distance-count"] == expected_pair_count
    assert metrics["metric-dp-distance-min"] > 0.0
    assert metrics["metric-dp-distance-invalid"] == 0.0


def test_diverged_models_fall_back_instead_of_raising() -> None:
    """A non-finite pairwise distance (diverged models) must not abort the round.

    Previously this raised, and since Strategy.start()'s Result accumulator
    lives inside the base-class call stack, an uncaught exception here
    silently discarded every prior round's history (see Phase 1 of
    docs/RESEARCH_ROADMAP.md -- 11/24 high-noise-multiplier runs failed this
    way with zero artifacts saved). It must instead fall back to a valid
    distance and flag the round as invalid so the run keeps producing a full
    history.
    """
    strategy = MetricPrivacyServerSideFixedClipping(
        strategy=FedAvg(),
        noise_multiplier=0.0,
        clipping_norm=10.0,
        num_sampled_clients=2,
    )
    strategy.current_arrays = model(np.array([0.0, 0.0]))

    replies = []
    for node_id, values in enumerate(([float("nan"), 0.0], [3.0, 4.0]), start=1):
        request = Message(
            content=RecordDict(), message_type="train", dst_node_id=node_id
        )
        content = RecordDict(
            {
                "arrays": model(np.asarray(values)),
                "metrics": MetricRecord({"num-examples": 1}),
            }
        )
        replies.append(Message(content=content, reply_to=request))

    arrays, metrics = strategy.aggregate_train(1, replies)

    assert arrays is not None
    assert metrics is not None
    assert np.isnan(metrics["metric-dp-distance"])
    assert metrics["metric-dp-distance-invalid"] == 1.0
    # No prior valid distance exists yet, so the fallback is 1.0.
    assert strategy.current_distance == pytest.approx(1.0)


def test_collapsed_zero_norm_updates_skip_round_instead_of_raising() -> None:
    """A round where every client's update collapses to zero must not abort.

    Flower's own DifferentialPrivacyServerSideFixedClipping.aggregate_train
    divides by each client update's L2 norm to compute a clipping scale,
    with no guard for an exactly-zero-norm update -- reachable when
    calibrated noise has already driven every client to return the current
    global model unchanged (observed live on real metric-privacy runs: a
    ZeroDivisionError from flwr.supercore.differential_privacy.
    clip_inputs_inplace crashed and lost a full training run's history).
    Must instead skip aggregation for the round (arrays=None keeps the
    strategy on the previous round's model) so the run keeps going.
    """
    strategy = MetricPrivacyServerSideFixedClipping(
        strategy=FedAvg(),
        noise_multiplier=0.0,
        clipping_norm=10.0,
        num_sampled_clients=2,
    )
    strategy.current_arrays = model(np.array([0.0, 0.0]))

    replies = []
    for node_id in (1, 2):
        request = Message(
            content=RecordDict(), message_type="train", dst_node_id=node_id
        )
        content = RecordDict(
            {
                # Identical to each other AND to current_arrays: no local
                # training signal at all, matching the observed divergence
                # state (pairwise distance among clients collapses to 0.0,
                # and each client's update relative to the previous global
                # model is also exactly 0.0).
                "arrays": model(np.array([0.0, 0.0])),
                "metrics": MetricRecord({"num-examples": 1}),
            }
        )
        replies.append(Message(content=content, reply_to=request))

    arrays, metrics = strategy.aggregate_train(1, replies)

    assert arrays is None
    assert metrics is not None
    assert metrics["metric-dp-aggregation-collapsed"] == 1.0
    assert metrics["metric-dp-distance-invalid"] == 1.0
