"""Tests for the global-DP Flower wrapper."""

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from metricdp_pytorch.globaldp_strategy import LoggedGlobalDPServerSideFixedClipping


def model(*arrays: np.ndarray) -> ArrayRecord:
    """Build a named ArrayRecord for a synthetic model."""
    return ArrayRecord(
        {f"layer-{index}": Array(array) for index, array in enumerate(arrays)}
    )


def test_collapsed_zero_norm_updates_skip_round_instead_of_raising() -> None:
    """A round where every client's update collapses to zero must not abort.

    Flower's own DifferentialPrivacyServerSideFixedClipping.aggregate_train
    divides by each client update's L2 norm to compute a clipping scale, with
    no guard for an exactly-zero-norm update. metricdp_strategy.py's
    MetricPrivacyServerSideFixedClipping already guards this same crash for
    its own route (calibrated noise blowing up); this class had no
    equivalent guard, reachable via sustained high global-dp noise instead
    (observed live sweeping --noise-multiplier up to 1.0 --
    results/noise_by_clients/sweep_progress.log logged 12 instances of this
    exact ZeroDivisionError, every one under global-dp). Must instead skip
    aggregation for the round (arrays=None keeps the strategy on the
    previous round's model) so the run keeps going.
    """
    strategy = LoggedGlobalDPServerSideFixedClipping(
        strategy=FedAvg(),
        noise_multiplier=1.0,
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
                # Identical to current_arrays: no local training signal at
                # all, matching the observed collapsed state (each client's
                # update relative to the previous global model is exactly
                # zero-norm).
                "arrays": model(np.array([0.0, 0.0])),
                "metrics": MetricRecord({"num-examples": 1}),
            }
        )
        replies.append(Message(content=content, reply_to=request))

    arrays, metrics = strategy.aggregate_train(1, replies)

    assert arrays is None
    assert metrics is not None
    assert metrics["global-dp-aggregation-collapsed"] == 1.0


def test_healthy_round_aggregates_normally() -> None:
    """A round with real client updates must aggregate and clear the flag."""
    strategy = LoggedGlobalDPServerSideFixedClipping(
        strategy=FedAvg(),
        noise_multiplier=0.05,
        clipping_norm=10.0,
        num_sampled_clients=2,
    )
    strategy.current_arrays = model(np.array([0.0, 0.0]))

    replies = []
    for node_id, value in ((1, 1.0), (2, 2.0)):
        request = Message(
            content=RecordDict(), message_type="train", dst_node_id=node_id
        )
        content = RecordDict(
            {
                "arrays": model(np.array([value, value])),
                "metrics": MetricRecord({"num-examples": 1}),
            }
        )
        replies.append(Message(content=content, reply_to=request))

    arrays, metrics = strategy.aggregate_train(1, replies)

    assert arrays is not None
    assert metrics is not None
    assert metrics["global-dp-aggregation-collapsed"] == 0.0
