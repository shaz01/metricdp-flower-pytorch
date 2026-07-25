"""Fast config-plumbing tests for the CIA runner (no real training)."""

from __future__ import annotations

from pathlib import Path

from experiments.cia.runner import (
    CIA_CLIPPING_NORM,
    CIA_LOCAL_EPOCHS,
    CIA_NOISE_MULTIPLIER,
    CIA_SEED,
    build_reproduce_command,
    run_name,
)


def test_run_name_is_stable_and_readable() -> None:
    assert run_name("vanilla", "fedavg") == "cia__vanilla__fedavg"
    assert run_name("metric-privacy", "fedyogi") == "cia__metric-privacy__fedyogi"


def test_build_reproduce_command_uses_cia_data_module_and_paper_settings() -> None:
    command = build_reproduce_command(
        privacy="global-dp",
        aggregation="fedprox",
        output_dir=Path("/tmp/cia-results"),
        max_parallel_clients=2,
    )
    joined = " ".join(command)
    assert "experiments.reproduce.runner" in joined
    assert "--data-module experiments.cia.dataset:create_cia_data_module" in joined
    assert "--num-clients 3" in joined
    assert "--rounds 1" in joined
    assert f"--local-epochs {CIA_LOCAL_EPOCHS}" in joined
    assert "--privacy global-dp" in joined
    assert "--aggregation fedprox" in joined
    assert f"--seed {CIA_SEED}" in joined
    assert f"--noise-multiplier {CIA_NOISE_MULTIPLIER}" in joined
    assert f"--clipping-norm {CIA_CLIPPING_NORM}" in joined
    assert "--run-name cia__global-dp__fedprox" in joined
    assert "--save-model" in joined
    assert "--max-parallel-clients 2" in joined
