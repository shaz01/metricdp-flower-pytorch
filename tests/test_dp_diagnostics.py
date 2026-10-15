"""Tests for metricdp_pytorch.dp_diagnostics."""

from __future__ import annotations

import numpy as np
from flwr.app import (
    Array,
    ArrayRecord,
    ConfigRecord,
    Error,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp.strategy import Strategy

from metricdp_pytorch.dp_diagnostics import (
    LeaveOneOutResidualLogging,
    clipping_diagnostics,
    leave_one_out_residual_diagnostics,
)


def model(*arrays: np.ndarray) -> ArrayRecord:
    """Build a named ArrayRecord for a synthetic model."""
    return ArrayRecord(
        {f"layer-{index}": Array(array) for index, array in enumerate(arrays)}
    )


def test_clipping_diagnostics_skips_content_free_error_replies() -> None:
    """A reply from a client that errored (e.g. a CUDA OOM mid-training) has no
    content -- reply.content raises ValueError in that case instead of
    returning None. clipping_diagnostics used to access .content unconditionally,
    crashing the whole server process on any single client failure during a
    global-dp round (found live on feature/cifar100-scaling: two CIA retry
    combos died this way within ~4 minutes of round 1, both with a
    ValueError("Message content is None...") traceback through this exact
    function; ported here since EuroSAT's global-dp combos share the same
    code path). It must instead skip the failed reply, the same way it
    already skips a non-ArrayRecord content, so a single bad client
    degrades the round instead of taking down the run.
    """
    request = Message(content=RecordDict(), message_type="train", dst_node_id=1)
    error_reply = Message(
        error=Error(code=1, reason="CUDA out of memory"), reply_to=request
    )

    good_request = Message(content=RecordDict(), message_type="train", dst_node_id=2)
    good_content = RecordDict(
        {
            "arrays": model(np.array([1.0, 1.0])),
            "metrics": MetricRecord({"client-id": 2}),
        }
    )
    good_reply = Message(content=good_content, reply_to=good_request)

    diagnostics = clipping_diagnostics(
        [error_reply, good_reply],
        current_arrays=model(np.array([0.0, 0.0])),
        clipping_norm=10.0,
    )

    assert diagnostics["dp-client-ids"] == [2]
    assert len(diagnostics["dp-update-norms-before-clipping"]) == 1


def test_clipping_diagnostics_all_replies_errored_returns_empty() -> None:
    """If every client in a round errored, diagnostics must degrade to empty
    results rather than crash -- matches the existing empty-clipped fallback
    (dp-fraction-clipped defaults to 0.0 when clipped is empty)."""
    request = Message(content=RecordDict(), message_type="train", dst_node_id=1)
    error_reply = Message(
        error=Error(code=1, reason="CUDA out of memory"), reply_to=request
    )

    diagnostics = clipping_diagnostics(
        [error_reply],
        current_arrays=model(np.array([0.0, 0.0])),
        clipping_norm=10.0,
    )

    assert diagnostics["dp-client-ids"] == []
    assert diagnostics["dp-fraction-clipped"] == 0.0


def train_reply(client_id: int, *arrays: np.ndarray) -> Message:
    """Build a successful train reply carrying a synthetic model."""
    request = Message(content=RecordDict(), message_type="train", dst_node_id=client_id + 1)
    content = RecordDict(
        {"arrays": model(*arrays), "metrics": MetricRecord({"client-id": client_id})}
    )
    return Message(content=content, reply_to=request)


def test_leave_one_out_residual_norms_match_hand_computation() -> None:
    """Updates u0=(3,0|1), u1=(0,0|1), u2=(0,3|1) around reference (1,1|0).

    Residual of client 0: u0 - mean(u1, u2) = (3,0,1) - (0,1.5,1) = (3,-1.5,0),
    norm sqrt(11.25); client 1: (0,0,1) - (1.5,1.5,1) -> sqrt(4.5);
    client 2 mirrors client 0.
    """
    reference = model(np.array([1.0, 1.0]), np.array([0.0]))
    replies = [
        train_reply(2, np.array([1.0, 4.0]), np.array([1.0])),
        train_reply(0, np.array([4.0, 1.0]), np.array([1.0])),
        train_reply(1, np.array([1.0, 1.0]), np.array([1.0])),
    ]

    diagnostics = leave_one_out_residual_diagnostics(replies, current_arrays=reference)

    assert diagnostics["loo-client-ids"] == [0, 1, 2]
    np.testing.assert_allclose(
        diagnostics["loo-residual-norms"],
        [np.sqrt(11.25), np.sqrt(4.5), np.sqrt(11.25)],
        rtol=1e-12,
    )


def test_leave_one_out_residual_skips_errors_and_needs_two_clients() -> None:
    request = Message(content=RecordDict(), message_type="train", dst_node_id=9)
    error_reply = Message(error=Error(code=1, reason="OOM"), reply_to=request)
    reference = model(np.array([0.0, 0.0]))

    single = leave_one_out_residual_diagnostics(
        [error_reply, train_reply(0, np.array([1.0, 0.0]))], current_arrays=reference
    )
    pair = leave_one_out_residual_diagnostics(
        [error_reply, train_reply(0, np.array([1.0, 0.0])), train_reply(1, np.array([0.0, 1.0]))],
        current_arrays=reference,
    )

    assert single == {"loo-client-ids": [], "loo-residual-norms": []}
    assert pair["loo-client-ids"] == [0, 1]
    np.testing.assert_allclose(pair["loo-residual-norms"], [np.sqrt(2.0)] * 2)


class RecordingStrategy(Strategy):
    """Inner strategy that records what it receives and clips replies in place."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def summary(self) -> None:
        pass

    def configure_train(self, server_round, arrays, config, grid):
        self.calls.append(("configure_train", server_round))
        return []

    def aggregate_train(self, server_round, replies):
        replies = list(replies)
        self.calls.append(("aggregate_train", len(replies)))
        for reply in replies:  # mimic Flower's in-place clipping to zero
            reply.content["arrays"] = model(np.zeros(2))
        return model(np.ones(2)), MetricRecord({"inner": 1.0})

    def configure_evaluate(self, server_round, arrays, config, grid):
        return []

    def aggregate_evaluate(self, server_round, replies):
        return MetricRecord({"evaluated": 1.0})


def test_wrapper_measures_before_inner_strategy_and_keeps_its_output() -> None:
    inner = RecordingStrategy()
    wrapper = LeaveOneOutResidualLogging(inner)
    reference = model(np.array([0.0, 0.0]))
    wrapper.configure_train(1, reference, ConfigRecord(), grid=None)

    arrays, metrics = wrapper.aggregate_train(
        1, iter([train_reply(0, np.array([2.0, 0.0])), train_reply(1, np.array([0.0, 0.0]))])
    )

    assert inner.calls == [("configure_train", 1), ("aggregate_train", 2)]
    assert arrays.to_numpy_ndarrays()[0].tolist() == [1.0, 1.0]
    assert metrics["inner"] == 1.0
    assert list(metrics["loo-client-ids"]) == [0, 1]
    np.testing.assert_allclose(list(metrics["loo-residual-norms"]), [2.0, 2.0])
    assert wrapper.aggregate_evaluate(1, []) == MetricRecord({"evaluated": 1.0})
