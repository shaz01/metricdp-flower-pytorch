"""Tests for selected-round server model checkpoints."""

from __future__ import annotations

import torch
from flwr.app import ArrayRecord, MetricRecord

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
