"""Tests for selected-round server model checkpoints."""

from __future__ import annotations

import pytest
import torch
from flwr.app import ArrayRecord, MetricRecord
from flwr.serverapp.strategy import Result

import experiments.reproduce.server as reproduce_server

from experiments.reproduce.server import (
    _checkpointing_evaluate_fn,
    checkpoint_model_path,
)


def test_checkpointing_evaluate_fn_saves_only_requested_rounds(tmp_path) -> None:
    calls: list[int] = []

    def evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        del arrays
        calls.append(server_round)
        return MetricRecord({"loss": float(server_round)})

    wrapped = _checkpointing_evaluate_fn(
        evaluate,
        output_dir=tmp_path,
        run_name="example",
        checkpoint_rounds={1, 3},
    )
    arrays = ArrayRecord({"weight": torch.tensor([2.0])})

    for round_number in range(4):
        wrapped(round_number, arrays)

    assert calls == [0, 1, 2, 3]
    assert not checkpoint_model_path(tmp_path, "example", 0).exists()
    assert checkpoint_model_path(tmp_path, "example", 1).exists()
    assert not checkpoint_model_path(tmp_path, "example", 2).exists()
    assert checkpoint_model_path(tmp_path, "example", 3).exists()
    state = torch.load(
        checkpoint_model_path(tmp_path, "example", 1),
        map_location="cpu",
        weights_only=True,
    )
    assert torch.equal(state["weight"], torch.tensor([2.0]))


def test_run_seeds_fallback_model_initialization(monkeypatch) -> None:
    class CapturingStrategy:
        def start(self, *, initial_arrays: ArrayRecord, **_kwargs) -> Result:
            return Result(arrays=initial_arrays)

    monkeypatch.setattr(
        reproduce_server,
        "create_paper_strategy",
        lambda **_kwargs: CapturingStrategy(),
    )
    monkeypatch.setattr(
        reproduce_server,
        "load_model",
        lambda _module: torch.nn.Linear(3, 2),
    )
    config = {
        "aggregation": "fedavg",
        "privacy": "vanilla",
        "num-clients": 2,
        "model-module": "unused",
        "learning-rate": 0.001,
        "num-server-rounds": 1,
        "seed": 17,
    }

    first = reproduce_server.run(None, config)
    torch.manual_seed(999)
    second = reproduce_server.run(None, config)

    first_state = first.arrays.to_torch_state_dict()
    second_state = second.arrays.to_torch_state_dict()
    assert first_state.keys() == second_state.keys()
    assert all(
        torch.equal(first_state[key], second_state[key]) for key in first_state
    )


def test_require_trained_arrays_raises_on_empty_result_arrays() -> None:
    """Flower's Strategy.start() only ever assigns result.arrays inside its
    aggregation-succeeded branch (flwr/serverapp/strategy/strategy.py's start():
    `if agg_arrays is not None: result.arrays = agg_arrays`) -- it never falls
    back to initial_arrays. If literally every round's aggregate_train call is
    skipped (e.g. DifferentialPrivacyServerSideFixedClipping discards the whole
    round on any client error, and every round hits a client error under
    sustained GPU pressure), result.arrays stays at Result()'s bare empty
    default for the entire run. Observed live: a CIA CIFAR-100 combo where all
    247 visible rounds logged "Some clients reported errors. Skipping
    aggregation." -- its leftover checkpoint was a genuine 0-key state_dict
    (2.8KB vs the ~18.5MB a real one is), which crashed model.load_state_dict()
    deep inside detailed_evaluation.py with a confusing "Missing key(s)"
    RuntimeError instead of a clear, actionable message.
    """
    from experiments.reproduce.server import _require_trained_arrays

    empty_result = Result()

    with pytest.raises(RuntimeError, match="ever successfully aggregated"):
        _require_trained_arrays(empty_result, run_name="some-run")


def test_require_trained_arrays_passes_on_nonempty_result_arrays() -> None:
    from experiments.reproduce.server import _require_trained_arrays

    result = Result()
    result.arrays = ArrayRecord({"weight": torch.tensor([1.0])})

    _require_trained_arrays(result, run_name="some-run")
