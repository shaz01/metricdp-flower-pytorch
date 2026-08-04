"""Tests for the end-to-end reproduction runner configuration."""

from __future__ import annotations

import pytest
import torch
from unittest.mock import patch

from experiments.reproduce.runner import _auto_client_gpus, _parser, build_run_config


def _args(*arguments: str):
    return _parser().parse_args(
        [
            "--seed",
            "42",
            "--noise-multiplier",
            "0.01",
            "--clipping-norm",
            "5.0",
            "--rounds",
            "20",
            "--local-epochs",
            "5",
            "--model-module",
            "experiments.reproduce.paper_cnn:create_model",
            *arguments,
        ]
    )


def test_default_runner_config_uses_paper_settings(tmp_path) -> None:
    config = build_run_config(
        _args("--output-dir", str(tmp_path), "--run-name", "test-run")
    )

    assert config["num-clients"] == 4
    assert config["num-server-rounds"] == 20
    assert config["local-epochs"] == 5
    assert config["learning-rate"] == 0.001
    assert config["partition-profile"] == "auto"
    assert config["data-module"] == (
        "experiments.reproduce.dataset.alzheimer:create_data_module"
    )
    assert config["model-module"] == (
        "experiments.reproduce.paper_cnn:create_model"
    )
    assert config["run-name"] == "test-run"


def test_smoke_runner_caps_work(tmp_path) -> None:
    config = build_run_config(_args("--smoke", "--output-dir", str(tmp_path)))

    assert config["num-server-rounds"] == 1
    assert config["local-epochs"] == 1
    assert config["initialization-epochs"] == 1
    assert config["max-client-samples"] == 32
    assert config["max-test-samples"] == 64


def test_custom_128_client_config_selects_scalable_auto_profile(tmp_path) -> None:
    config = build_run_config(
        _args(
            "--num-clients",
            "128",
            "--partition",
            "non-iid",
            "--output-dir",
            str(tmp_path),
        )
    )

    assert config["num-clients"] == 128
    assert config["partition-profile"] == "auto"


def test_runner_passes_plugin_specific_partition_values_through(tmp_path) -> None:
    config = build_run_config(
        _args(
            "--partition",
            "custom-label-skew",
            "--partition-profile",
            "dataset-profile-a",
            "--data-module",
            "my_package.data:create_data_module",
            "--model-module",
            "my_package.model:create_model",
            "--output-dir",
            str(tmp_path),
        )
    )

    assert config["partition-mode"] == "custom-label-skew"
    assert config["partition-profile"] == "dataset-profile-a"
    assert config["data-module"] == "my_package.data:create_data_module"
    assert config["model-module"] == "my_package.model:create_model"


def test_runner_passes_checkpoint_rounds_through(tmp_path) -> None:
    config = build_run_config(
        _args(
            "--checkpoint-rounds",
            "1",
            "20",
            "--output-dir",
            str(tmp_path),
        )
    )

    assert config["checkpoint-rounds"] == [1, 20]


def test_runner_rejects_checkpoint_after_final_round(tmp_path) -> None:
    with pytest.raises(ValueError, match="checkpoint-rounds"):
        build_run_config(
            _args(
                "--checkpoint-rounds",
                "21",
                "--output-dir",
                str(tmp_path),
            )
        )


def test_runner_passes_cia_data_module_options_through(tmp_path) -> None:
    config = build_run_config(
        _args(
            "--num-clients",
            "16",
            "--target-partition-id",
            "7",
            "--shadow-fraction",
            "0.2",
            "--train-fraction",
            "0.75",
            "--output-dir",
            str(tmp_path),
        )
    )

    assert config["target-partition-id"] == 7
    assert config["shadow-fraction"] == 0.2
    assert config["train-fraction"] == 0.75


def test_auto_client_gpus_is_zero_without_cuda() -> None:
    """CPU and Apple-MPS hosts keep the historical CPU-only client behaviour."""
    with patch.object(torch.cuda, "is_available", return_value=False):
        assert _auto_client_gpus(num_clients=48, max_parallel_clients=12) == 0.0


@pytest.mark.parametrize(
    ("num_clients", "max_parallel_clients"),
    [(48, 12), (48, 4), (48, 16), (4, 4), (48, 48), (2, 100)],
)
def test_auto_client_gpus_never_oversubscribes_the_device(
    num_clients: int, max_parallel_clients: int
) -> None:
    """Concurrent actors must always fit on a single GPU, despite float rounding."""
    with patch.object(torch.cuda, "is_available", return_value=True):
        share = _auto_client_gpus(
            num_clients=num_clients, max_parallel_clients=max_parallel_clients
        )

    concurrent = max(1, min(num_clients, max_parallel_clients))
    assert share > 0.0
    assert share * concurrent <= 1.0
