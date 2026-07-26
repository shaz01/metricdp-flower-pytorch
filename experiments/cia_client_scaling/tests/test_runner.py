"""Fast config-plumbing tests for the 48-client CIA runner (no real training)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from experiments.cia_client_scaling import runner as runner_module
from experiments.cia_client_scaling.runner import (
    SEED,
    TIMING_CONFIGS,
    build_reproduce_command,
    is_training_complete,
    run_name,
    run_one_combo,
)


def test_timing_configs_match_design_spec() -> None:
    assert TIMING_CONFIGS["first-round"] == {
        "rounds": 1,
        "local_epochs": 20,
        "noise_multiplier": 0.01,
        "clipping_norm": 5.0,
    }
    assert TIMING_CONFIGS["post-convergence"] == {
        "rounds": 20,
        "local_epochs": 5,
        "noise_multiplier": 0.05,
        "clipping_norm": 5.0,
    }


def test_run_name_is_stable_and_readable() -> None:
    assert (
        run_name("homogeneous", "first-round", "vanilla", "fedavg")
        == "cia_scaling__first-round__homogeneous__vanilla__fedavg"
    )
    assert (
        run_name("non-iid", "post-convergence", "metric-privacy", "fedyogi")
        == "cia_scaling__post-convergence__non-iid__metric-privacy__fedyogi"
    )


def test_build_reproduce_command_first_round(tmp_path: Path) -> None:
    command = build_reproduce_command(
        partition_mode="homogeneous",
        timing="first-round",
        privacy="global-dp",
        aggregation="fedyogi",
        output_dir=tmp_path,
        max_parallel_clients=4,
    )
    joined = " ".join(str(part) for part in command)
    assert "experiments.reproduce.runner" in joined
    assert "--num-clients 48" in joined
    assert "--partition homogeneous" in joined
    assert "--privacy global-dp" in joined
    assert "--aggregation fedyogi" in joined
    assert "--rounds 1" in joined
    assert "--local-epochs 20" in joined
    assert "--noise-multiplier 0.01" in joined
    assert "--clipping-norm 5.0" in joined
    assert f"--seed {SEED}" in joined
    assert (
        "--run-name cia_scaling__first-round__homogeneous__global-dp__fedyogi"
        in joined
    )
    assert "--save-model" in joined
    assert "--max-parallel-clients 4" in joined


def test_build_reproduce_command_post_convergence(tmp_path: Path) -> None:
    command = build_reproduce_command(
        partition_mode="non-iid",
        timing="post-convergence",
        privacy="vanilla",
        aggregation="fedavg",
        output_dir=tmp_path,
        max_parallel_clients=4,
    )
    joined = " ".join(str(part) for part in command)
    assert "--rounds 20" in joined
    assert "--local-epochs 5" in joined
    assert "--noise-multiplier 0.05" in joined
    assert (
        "--run-name cia_scaling__post-convergence__non-iid__vanilla__fedavg"
        in joined
    )


def test_is_training_complete_true_when_expected_rounds_present(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps({"server_evaluate_metrics": {"0": {}, "1": {}, "2": {}}})
    )
    assert is_training_complete(path, expected_rounds=2)


def test_is_training_complete_false_when_missing(tmp_path: Path) -> None:
    assert not is_training_complete(tmp_path / "missing.json", expected_rounds=1)


def test_is_training_complete_false_when_short_of_rounds(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"server_evaluate_metrics": {"0": {}, "1": {}}}))
    assert not is_training_complete(path, expected_rounds=5)


def test_is_training_complete_false_on_unparseable_json(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("not json")
    assert not is_training_complete(path, expected_rounds=1)


def test_run_one_combo_skips_subprocess_when_already_complete(
    tmp_path: Path, monkeypatch
) -> None:
    name = run_name("homogeneous", "first-round", "vanilla", "fedavg")
    result_path = tmp_path / f"{name}.json"
    result_path.write_text(json.dumps({"server_evaluate_metrics": {"0": {}, "1": {}}}))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called when already complete")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    model_path, success = run_one_combo(
        partition_mode="homogeneous",
        timing="first-round",
        privacy="vanilla",
        aggregation="fedavg",
        output_dir=tmp_path,
        max_parallel_clients=4,
        force=False,
    )
    assert success is True
    assert model_path == tmp_path / f"{name}.pt"


def test_run_one_combo_runs_subprocess_when_incomplete(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[list[str]] = []

    class FakeCompletedProcess:
        returncode = 0

    def fake_run(command, cwd=None):
        calls.append(command)
        return FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)

    _model_path, success = run_one_combo(
        partition_mode="non-iid",
        timing="post-convergence",
        privacy="metric-privacy",
        aggregation="fedyogi",
        output_dir=tmp_path,
        max_parallel_clients=4,
        force=False,
    )
    assert success is True
    assert len(calls) == 1
    assert "--rounds 20" in " ".join(calls[0])


def test_run_one_combo_reports_failure_on_nonzero_returncode(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeFailedProcess:
        returncode = 1

    monkeypatch.setattr(subprocess, "run", lambda command, cwd=None: FakeFailedProcess())

    _model_path, success = run_one_combo(
        partition_mode="homogeneous",
        timing="first-round",
        privacy="vanilla",
        aggregation="fedavg",
        output_dir=tmp_path,
        max_parallel_clients=4,
        force=False,
    )
    assert success is False


def test_run_cia_client_scaling_continues_past_failed_combo_and_writes_report(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeCompletedProcess:
        returncode = 0

    monkeypatch.setattr(
        subprocess, "run", lambda command, cwd=None: FakeCompletedProcess()
    )

    def fake_evaluate_combo(model_path, *, partition_mode, device):
        if "fedyogi" in model_path.name:
            raise RuntimeError("boom")
        return 0.5, 0.6

    monkeypatch.setattr(runner_module, "evaluate_combo", fake_evaluate_combo)

    results = runner_module.run_cia_client_scaling(
        output_dir=tmp_path,
        partition_modes=("homogeneous",),
        privacy_modes=("vanilla",),
        aggregations=("fedavg", "fedyogi"),
        timings=("first-round",),
        max_parallel_clients=4,
        force=False,
    )

    assert len(results) == 1
    assert results[0].aggregation == "fedavg"

    failed_name = run_name("homogeneous", "first-round", "vanilla", "fedyogi")
    log_content = (tmp_path / "sweep_progress.log").read_text()
    assert failed_name in log_content
    assert "evaluation error" in log_content

    report_path = tmp_path / "cia_client_scaling.json"
    assert report_path.exists()
    report_data = json.loads(report_path.read_text())
    assert len(report_data) == 1
    assert report_data[0]["aggregation"] == "fedavg"
