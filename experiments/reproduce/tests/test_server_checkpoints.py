"""Tests for selected-round server model checkpoints."""

from __future__ import annotations

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
