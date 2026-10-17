"""G-M0 noise-independence check (CIA roadmap v2, Section 2 item 7): real 2-round Flower runs.

Four global-DP runs of a small EuroSAT federation on the CPU. Each run logs, per round, the
realized noise energy ``||Delta_b||^2 - ||s_b||^2 - E||xi_b||^2`` on 64 fixed coordinate blocks
(``--log-noise-diagnostics``), the statistic of ``research/cia_roadmap_v2/scripts/measurement_audit.py``
per block, so each run gives 2 x 64 = 128 values (null standard deviation of a correlation about 0.09).

- equal seeds: correlation above 0.9;
- equal noise seed, different client seed (the v1 coupling): above 0.9, and the trained models differ;
- different noise and client seeds (independent worlds): within [-0.3, 0.3].

The noise std (0.2 per coordinate) makes the noise dominate the block statistic, as it does
at the noise levels the roadmap studies; the frontier noise is far smaller only in absolute terms.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[4]
CLIENTS = 4
ROUNDS = 2
NOISE_STD = 0.2
CLIPPING_NORM = 5.0
RUNS = {"reference": (1, 1), "equal": (1, 1), "client-only": (1, 2), "independent": (2, 2)}  # (noise, client)


def _eurosat_cached() -> bool:
    try:
        from experiments.reproduce.dataset.eurosat import load_eurosat_dataset
        load_eurosat_dataset()
    except Exception:
        return False
    return True


def _train(output: Path, name: str, noise_seed: int, client_seed: int) -> dict:
    command = [
        sys.executable, "-m", "experiments.reproduce.runner",
        "--data-module", "experiments.reproduce.dataset.eurosat:create_data_module",
        "--model-module", "experiments.reproduce.eurosat_cnn:create_model",
        "--privacy", "global-dp", "--aggregation", "fedavg", "--aggregation-weighting", "equal",
        "--num-clients", str(CLIENTS), "--rounds", str(ROUNDS), "--local-epochs", "1",
        "--seed", "42", "--noise-seed", str(noise_seed), "--client-seed", str(client_seed),
        "--noise-multiplier", str(NOISE_STD * CLIENTS / CLIPPING_NORM), "--clipping-norm", str(CLIPPING_NORM),
        "--initialization-epochs", "1", "--max-client-samples", "64", "--max-test-samples", "64",
        "--log-noise-diagnostics", "--save-model", "--max-parallel-clients", str(CLIENTS),
        "--output-dir", str(output), "--run-name", name,
    ]
    env = os.environ | {"CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT)}
    subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    return json.loads((output / f"{name}.json").read_text())


def _blocks(run: dict) -> np.ndarray:
    metrics = run["train_metrics"]
    return np.concatenate([metrics[str(r)]["dp-noise-energy-blocks"] for r in range(1, ROUNDS + 1)])


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    if not _eurosat_cached():
        pytest.skip("EuroSAT is not available locally")
    output = tmp_path_factory.mktemp("noise-independence")
    return {name: (_train(output, name, *seeds), output / f"{name}.pt") for name, seeds in RUNS.items()}


def _corr(runs, name):
    value = float(np.corrcoef(_blocks(runs["reference"][0]), _blocks(runs[name][0]))[0, 1])
    print(f"realized-noise-energy correlation, reference vs {name}: {value:.4f}")
    return value


def test_runs_log_the_seeds_and_128_block_values(runs):
    for name, (noise_seed, client_seed) in RUNS.items():
        run = runs[name][0]
        assert run["metadata"]["noise_seed"] == noise_seed
        assert run["metadata"]["client_seed"] == client_seed
        assert _blocks(run).shape == (ROUNDS * 64,)


def test_equal_seeds_give_correlated_noise(runs):
    assert _corr(runs, "equal") > 0.9


def test_equal_noise_seed_couples_the_noise_across_client_seeds(runs):
    assert _corr(runs, "client-only") > 0.9


def test_client_seed_changes_training(runs):
    import torch
    reference = torch.load(runs["reference"][1], weights_only=True)
    other = torch.load(runs["client-only"][1], weights_only=True)
    assert any(not torch.equal(reference[key], other[key]) for key in reference)


def test_different_noise_seeds_give_independent_noise(runs):
    assert -0.3 <= _corr(runs, "independent") <= 0.3
