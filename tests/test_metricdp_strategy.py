"""Tests for metric-aware DP distance calibration."""

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from metricdp_pytorch.metricdp_strategy import MetricPrivacyServerSideFixedClipping


def model(*arrays: np.ndarray) -> ArrayRecord:
    """Build a named ArrayRecord for a synthetic model."""
    return ArrayRecord(
        {f"layer-{index}": Array(array) for index, array in enumerate(arrays)}
    )


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
    assert metrics["metric-dp-distance-median"] == pytest.approx(np.sqrt(20.0))
    assert metrics["metric-dp-distance-mean"] == pytest.approx(np.sqrt(20.0))
    assert metrics["metric-dp-noise-stdv"] == 0.0
    assert metrics["dp-update-norms-before-clipping"] == pytest.approx([1.0, 5.0])
    assert metrics["dp-client-clipped"] == [0, 0]
    assert metrics["dp-fraction-clipped"] == 0.0
    assert metrics["dp-parameter-count"] == 2
    assert metrics["dp-expected-noise-l2-norm"] == 0.0
