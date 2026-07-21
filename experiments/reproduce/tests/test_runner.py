"""Tests for the end-to-end reproduction runner configuration."""

from __future__ import annotations

from experiments.reproduce.runner import _parser, build_run_config


def _args(*arguments: str):
    return _parser().parse_args(list(arguments))


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
            "--output-dir",
            str(tmp_path),
        )
    )

    assert config["partition-mode"] == "custom-label-skew"
    assert config["partition-profile"] == "dataset-profile-a"
    assert config["data-module"] == "my_package.data:create_data_module"
