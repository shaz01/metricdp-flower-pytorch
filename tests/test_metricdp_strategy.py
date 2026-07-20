"""Tests for metric-aware DP distance calibration."""

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from metricdp_pytorch.metricdp_strategy import (
    MetricPrivacyServerSideFixedClipping,
    maximum_pairwise_model_distance,
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
    assert metrics["metric-dp-noise-stdv"] == 0.0
