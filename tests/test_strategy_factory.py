"""Tests for the aggregation/privacy strategy factory."""

from __future__ import annotations

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict

from metricdp_pytorch.strategy_factory import (
    FEDOPT_ETA_0,
    FEDOPT_ETA_DECAY,
    DecayingEtaFedAdam,
    make_base_strategy,
)


def _fit_reply(values: dict[str, np.ndarray]) -> Message:
    content = RecordDict(
        {
            "arrays": ArrayRecord({key: Array(value) for key, value in values.items()}),
            "metrics": MetricRecord({"num-examples": 1}),
        }
    )
    request = Message(content=RecordDict({}), dst_node_id=0, message_type="train")
    return Message(content=content, reply_to=request)


def test_decaying_eta_fedadam_shrinks_eta_each_round() -> None:
    """eta_t = eta_0 / (1 + decay * (round - 1)), matching the schedule
    verified via 20 real federated rounds to resolve FedOpt's late-round
    oscillation (see strategy_factory.make_base_strategy's docstring)."""
    strategy = DecayingEtaFedAdam(
        eta_0=0.01, decay=0.15, beta_1=0.0, beta_2=0.0, tau=1e-3
    )
    strategy.current_arrays = {"layer-0": np.zeros(2, dtype=np.float32)}

    strategy.aggregate_train(1, [_fit_reply({"layer-0": np.array([1.0, 1.0])})])
    assert strategy.eta == pytest.approx(0.01)

    strategy.current_arrays = {"layer-0": np.zeros(2, dtype=np.float32)}
    strategy.aggregate_train(11, [_fit_reply({"layer-0": np.array([1.0, 1.0])})])
    assert strategy.eta == pytest.approx(0.01 / 2.5)


def test_make_base_strategy_fedopt_uses_the_verified_decay_schedule() -> None:
    strategy = make_base_strategy("fedopt", num_clients=4)
    assert isinstance(strategy, DecayingEtaFedAdam)
    assert strategy._eta_0 == FEDOPT_ETA_0
    assert strategy._decay == FEDOPT_ETA_DECAY
