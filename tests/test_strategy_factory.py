"""Tests for the aggregation/privacy strategy factory."""

from __future__ import annotations

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict

from metricdp_pytorch.strategy_factory import (
    FEDOPT_ETA_0,
    FEDOPT_ETA_DECAY,
    DecayingEtaFedAdam,
    DeterministicReplyOrderMixin,
    make_base_strategy,
)


def _fit_reply(
    values: dict[str, np.ndarray], *, client_id: int = 0
) -> Message:
    content = RecordDict(
        {
            "arrays": ArrayRecord({key: Array(value) for key, value in values.items()}),
            "metrics": MetricRecord(
                {"client-id": client_id, "num-examples": 1}
            ),
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


@pytest.mark.parametrize(
    "aggregation",
    ["fedavg", "fedavgm", "fedmedian", "fedprox", "fedopt", "fedyogi"],
)
def test_all_base_strategies_sort_train_replies_by_client_id(
    aggregation: str,
) -> None:
    strategy = make_base_strategy(aggregation, num_clients=3)
    assert isinstance(strategy, DeterministicReplyOrderMixin)


def test_fedavg_aggregation_is_independent_of_reply_arrival_order() -> None:
    replies = [
        _fit_reply({"layer-0": np.array([1e20], dtype=np.float32)}, client_id=0),
        _fit_reply({"layer-0": np.array([-1e20], dtype=np.float32)}, client_id=1),
        _fit_reply({"layer-0": np.array([1.0], dtype=np.float32)}, client_id=2),
    ]
    forward = make_base_strategy("fedavg", num_clients=3)
    reordered = make_base_strategy("fedavg", num_clients=3)

    forward_arrays, _ = forward.aggregate_train(1, replies)
    reordered_arrays, _ = reordered.aggregate_train(
        1, [replies[0], replies[2], replies[1]]
    )

    assert forward_arrays is not None
    assert reordered_arrays is not None
    assert np.array_equal(
        forward_arrays.to_numpy_ndarrays()[0],
        reordered_arrays.to_numpy_ndarrays()[0],
    )
