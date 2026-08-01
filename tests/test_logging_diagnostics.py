"""Tests for persisted DP and aligned per-client diagnostics."""

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict
from flwr.serverapp.strategy import FedAvg

from metricdp_pytorch.globaldp_strategy import LoggedGlobalDPServerSideFixedClipping
from metricdp_pytorch.metrics import aggregate_metrics_with_clients


def _model(values) -> ArrayRecord:
    return ArrayRecord({"weights": Array(np.asarray(values, dtype=np.float64))})


def _reply(client_id: int, values, *, loss: float) -> Message:
    request = Message(
        content=RecordDict(), message_type="train", dst_node_id=client_id + 1
    )
    return Message(
        content=RecordDict(
            {
                "arrays": _model(values),
                "metrics": MetricRecord(
                    {
                        "client-id": client_id,
                        "train_loss": loss,
                        "num-examples": client_id + 1,
                    }
                ),
            }
        ),
        reply_to=request,
    )


def test_metric_aggregation_preserves_aligned_client_values() -> None:
    records = [
        _reply(7, [0.0], loss=0.7).content,
        _reply(3, [0.0], loss=0.3).content,
    ]

    metrics = aggregate_metrics_with_clients(records, "num-examples")

    assert metrics["client-ids"] == [3, 7]
    assert metrics["per-client-train_loss"] == [0.3, 0.7]
    assert metrics["per-client-num-examples"] == [4, 8]
    assert metrics["train_loss"] == pytest.approx((0.3 * 4 + 0.7 * 8) / 12)


def test_global_dp_logs_sigma_clipping_and_signal_diagnostics() -> None:
    strategy = LoggedGlobalDPServerSideFixedClipping(
        strategy=FedAvg(train_metrics_aggr_fn=aggregate_metrics_with_clients),
        noise_multiplier=0.0,
        clipping_norm=2.0,
        num_sampled_clients=2,
    )
    strategy.current_arrays = _model([0.0, 0.0])

    arrays, metrics = strategy.aggregate_train(
        1,
        [_reply(1, [3.0, 4.0], loss=0.7), _reply(0, [1.0, 0.0], loss=0.3)],
    )

    assert arrays is not None
    assert metrics is not None
    assert metrics["global-dp-noise-stdv"] == 0.0
    assert metrics["dp-client-ids"] == [0, 1]
    assert metrics["dp-update-norms-before-clipping"] == pytest.approx([1.0, 5.0])
    assert metrics["dp-client-clipped"] == [0, 1]
    assert metrics["dp-num-clients-clipped"] == 1
    assert metrics["dp-fraction-clipped"] == 0.5
    assert metrics["dp-signal-update-norm"] > 0.0
    assert metrics["dp-parameter-count"] == 2
    assert metrics["dp-expected-noise-l2-norm"] == 0.0
    assert metrics["client-ids"] == [0, 1]
    assert metrics["per-client-train_loss"] == [0.3, 0.7]
