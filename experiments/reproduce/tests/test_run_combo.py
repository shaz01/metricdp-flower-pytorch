"""Checkpoint-round validation of run_one_combo (round 0 = the initial global model)."""

from __future__ import annotations

import subprocess

import pytest

from experiments.reproduce.matrix import Combo, Hyperparams
from experiments.reproduce.matrix import run_combo


def _combo() -> Combo:
    return Combo(
        name_prefix="t", num_clients=3, partition="homogeneous", privacy="vanilla",
        aggregation="fedavg", seed=42, noise_multiplier=0.01,
        hyperparams=Hyperparams(clipping_norm=5.0, rounds=2, local_epochs=1, batch_size=32,
                                learning_rate=0.001, initialization_epochs=1),
        data_module="experiments.reproduce.dataset.alzheimer:create_data_module",
        model_module="experiments.reproduce.paper_cnn:create_model",
    )


def test_round_zero_checkpoint_is_passed_to_the_runner(tmp_path, monkeypatch) -> None:
    commands = []
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command)
                        or subprocess.CompletedProcess(command, 0))
    assert run_combo.run_one_combo(_combo(), output_dir=tmp_path, max_parallel_clients=1,
                                   force=True, log=lambda _: None, checkpoint_rounds=(0, 1, 2))
    command = commands[0]
    index = command.index("--checkpoint-rounds")
    assert command[index + 1:index + 4] == ["0", "1", "2"]


@pytest.mark.parametrize("rounds", [(-1,), (3,)])
def test_out_of_range_checkpoint_rounds_are_rejected(tmp_path, rounds) -> None:
    with pytest.raises(ValueError, match="between 0 and combo rounds"):
        run_combo.run_one_combo(_combo(), output_dir=tmp_path, max_parallel_clients=1,
                                force=True, log=lambda _: None, checkpoint_rounds=rounds)
