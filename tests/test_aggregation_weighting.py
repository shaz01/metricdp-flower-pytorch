"""Equal vs size-weighted FedAvg aggregation (``--aggregation-weighting``)."""

from __future__ import annotations

import numpy as np
import pytest
from flwr.app import Array, ArrayRecord, Message, MetricRecord, RecordDict

from experiments.reproduce.client import _weight_metrics
from experiments.reproduce.matrix.combo import Combo
from experiments.reproduce.matrix.hyperparams import Hyperparams
from metricdp_pytorch.strategy_factory import (
    UNIT_WEIGHT_KEY,
    aggregation_token,
    make_base_strategy,
    make_strategy,
    weighting_key,
)

SIZES = (10, 30, 160)
VALUES = (np.array([1.0, 0.0]), np.array([0.0, 2.0]), np.array([30.0, 40.0]))  # norms 1, 2, 50


def _model(value: np.ndarray) -> ArrayRecord:
    return ArrayRecord({"layer-0": Array(np.asarray(value, dtype=np.float64))})


def _replies(weighting: str, *, evaluate: bool = False) -> list[Message]:
    config = {"aggregation-weighting": weighting}
    out = []
    for cid, (size, value) in enumerate(zip(SIZES, VALUES, strict=True)):
        metrics = MetricRecord({"client-id": cid, "num-examples": size, "loss": float(cid),
                                **_weight_metrics(config)})
        content = {"metrics": metrics} if evaluate else {"arrays": _model(value), "metrics": metrics}
        request = Message(content=RecordDict(), dst_node_id=cid + 1,
                          message_type="evaluate" if evaluate else "train")
        out.append(Message(content=RecordDict(content), reply_to=request))
    return out


def _aggregate(strategy, weighting):
    arrays, metrics = strategy.aggregate_train(1, _replies(weighting))
    return arrays.to_numpy_ndarrays()[0], metrics


def test_weighting_key_and_run_name_token():
    assert weighting_key("num-examples") == "num-examples"
    assert weighting_key("equal") == UNIT_WEIGHT_KEY
    assert aggregation_token("fedavg", "num-examples") == "fedavg"
    assert aggregation_token("fedavg", "equal") == "fedavg-eqw"
    with pytest.raises(ValueError):
        weighting_key("size")


def test_client_reports_unit_weight_only_in_equal_mode():
    assert _weight_metrics({}) == {}
    assert _weight_metrics({"aggregation-weighting": "num-examples"}) == {}
    assert _weight_metrics({"aggregation-weighting": "equal"}) == {UNIT_WEIGHT_KEY: 1}


def test_default_mode_is_size_weighted_and_unchanged():
    default = make_base_strategy("fedavg", num_clients=3)
    assert default.weighted_by_key == "num-examples"
    aggregate, _ = _aggregate(default, "num-examples")
    expected = sum(s * v for s, v in zip(SIZES, VALUES)) / sum(SIZES)
    np.testing.assert_allclose(aggregate, expected)


def test_equal_mode_is_plain_mean():
    strategy = make_base_strategy("fedavg", num_clients=3, weighting="equal")
    assert strategy.weighted_by_key == UNIT_WEIGHT_KEY
    aggregate, metrics = _aggregate(strategy, "equal")
    np.testing.assert_allclose(aggregate, np.mean(VALUES, axis=0))
    assert metrics["loss"] == pytest.approx(1.0)  # train metrics equal-weighted too


def test_equal_mode_keeps_client_evaluation_example_weighted():
    strategy = make_base_strategy("fedavg", num_clients=3, weighting="equal")
    metrics = strategy.aggregate_evaluate(1, _replies("equal", evaluate=True))
    assert metrics["loss"] == pytest.approx((0 * 10 + 1 * 30 + 2 * 160) / 200)


@pytest.mark.parametrize("privacy", ["global-dp", "metric-privacy"])
def test_equal_mode_dp_aggregate_is_mean_of_clipped_updates(privacy):
    clip = 5.0
    strategy = make_strategy("fedavg", privacy, num_clients=3, fraction_evaluate=1.0,
                             noise_multiplier=0.0, clipping_norm=clip, weighting="equal")
    strategy.current_arrays = _model(np.zeros(2))
    aggregate, _ = _aggregate(strategy, "equal")
    clipped = [v * min(1.0, clip / np.linalg.norm(v)) for v in VALUES]
    np.testing.assert_allclose(aggregate, np.mean(clipped, axis=0), atol=1e-12)


@pytest.mark.parametrize("privacy", ["global-dp", "metric-privacy"])
def test_dp_noise_stdv_does_not_depend_on_weighting(privacy):
    stdvs = {}
    for weighting in ("num-examples", "equal"):
        strategy = make_strategy("fedavg", privacy, num_clients=3, fraction_evaluate=1.0,
                                 noise_multiplier=0.3, clipping_norm=5.0, weighting=weighting)
        strategy.current_arrays = _model(np.zeros(2))
        _, metrics = _aggregate(strategy, weighting)
        key = "global-dp-noise-stdv" if privacy == "global-dp" else "metric-dp-noise-stdv"
        stdvs[weighting] = metrics[key]
    assert stdvs["equal"] == stdvs["num-examples"]
    if privacy == "global-dp":
        assert stdvs["equal"] == pytest.approx(0.3 * 5.0 / 3)


def _combo(**kwargs) -> Combo:
    return Combo(name_prefix="x", num_clients=4, partition="homogeneous", privacy="vanilla",
                 aggregation="fedavg", seed=42, noise_multiplier=0.0,
                 hyperparams=Hyperparams(clipping_norm=5.0, rounds=1, local_epochs=1, batch_size=32,
                                             learning_rate=0.001, initialization_epochs=0), data_module="a.b:c",
                 model_module="d.e:f", **kwargs)


def test_combo_run_name_and_runner_args():
    default, equal = _combo(), _combo(aggregation_weighting="equal")
    assert "__fedavg__" in default.run_name()
    assert "__fedavg-eqw__" in equal.run_name()
    args = dict(output_dir="o", max_parallel_clients=1, client_cpus=1.0)
    assert "--aggregation-weighting" not in default.runner_args(**args)
    tail = equal.runner_args(**args)
    assert tail[tail.index("--aggregation-weighting") + 1] == "equal"
