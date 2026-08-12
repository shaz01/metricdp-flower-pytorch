"""Tests for metricdp_pytorch.dp_diagnostics."""

from __future__ import annotations

import numpy as np
from flwr.app import Array, ArrayRecord, Error, Message, MetricRecord, RecordDict

from metricdp_pytorch.dp_diagnostics import clipping_diagnostics


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
    global-dp round (observed live: two CIA CIFAR-100 retry combos died this way
    within ~4 minutes of round 1, both with a ValueError("Message content is
    None...") traceback through this exact function). It must instead skip the
    failed reply, the same way it already skips a non-ArrayRecord content, so a
    single bad client degrades the round instead of taking down the run.
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
