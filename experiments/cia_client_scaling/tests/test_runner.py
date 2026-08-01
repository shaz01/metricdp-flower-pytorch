"""Fast config-plumbing tests for the 48-client CIA runner (no real training)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from experiments.cia_client_scaling import runner as runner_module
from experiments.cia_client_scaling.runner import (
    AGGREGATIONS,
    PARTITION_MODES,
    PRIVACY_MODES,
    SEED,
    TIMING_CONFIGS,
    _parser,
    build_combo,
    main,
    parse_subset,
    resolve_noise_multiplier,
)
from experiments.reproduce.matrix import is_complete, run_one_combo


def _combo_name(
    partition_mode: str,
    timing: str,
    privacy: str,
    aggregation: str,
    *,
    noise_multiplier: float | None = None,
) -> str:
    return build_combo(
        partition_mode=partition_mode,
        timing=timing,
        privacy=privacy,
        aggregation=aggregation,
        noise_multiplier=noise_multiplier,
    ).run_name()


def _run_training_combo(
    *,
    partition_mode: str,
    timing: str,
    privacy: str,
    aggregation: str,
    output_dir: Path,
    max_parallel_clients: int,
    force: bool,
) -> tuple[Path, bool]:
    combo = build_combo(
        partition_mode=partition_mode,
        timing=timing,
        privacy=privacy,
        aggregation=aggregation,
    )
    model_path = output_dir / f"{combo.run_name()}.pt"
    success = run_one_combo(
        combo,
        output_dir=output_dir,
        max_parallel_clients=max_parallel_clients,
        force=force,
        log=lambda _message: None,
        save_model=True,
    )
    return model_path, success


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


def test_combo_includes_explicit_data_module_in_run_name() -> None:
    assert _combo_name("homogeneous", "first-round", "vanilla", "fedavg") == (
        "cia_scaling__first-round__homogeneous__vanilla__fedavg__clients-48__"
        "seed-42__nm0p01__clip5__rounds-1__epochs-20__alzheimer"
    )


def test_build_combo_first_round(tmp_path: Path) -> None:
    combo = build_combo(
        partition_mode="homogeneous",
        timing="first-round",
        privacy="global-dp",
        aggregation="fedyogi",
    )
    joined = " ".join(
        combo.runner_args(
            output_dir=tmp_path,
            max_parallel_clients=4,
            client_cpus=1.0,
            save_model=True,
        )
    )
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
        "--run-name cia_scaling__first-round__homogeneous__global-dp__fedyogi__"
        "clients-48__seed-42__nm0p01__clip5__rounds-1__epochs-20" in joined
    )
    assert "--save-model" in joined
    assert "--max-parallel-clients 4" in joined


def test_build_combo_post_convergence(tmp_path: Path) -> None:
    combo = build_combo(
        partition_mode="non-iid",
        timing="post-convergence",
        privacy="vanilla",
        aggregation="fedavg",
    )
    joined = " ".join(
        combo.runner_args(output_dir=tmp_path, max_parallel_clients=4, client_cpus=1.0)
    )
    assert "--rounds 20" in joined
    assert "--local-epochs 5" in joined
    assert "--noise-multiplier 0.05" in joined
    assert (
        "--run-name cia_scaling__post-convergence__non-iid__vanilla__fedavg__"
        "clients-48__seed-42__nm0p05__clip5__rounds-20__epochs-5" in joined
    )


def test_is_training_complete_true_when_expected_rounds_present(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    path.write_text(
        json.dumps({"server_evaluate_metrics": {"0": {}, "1": {}, "2": {}}})
    )
    assert is_complete(path, expected_rounds=2)


def test_is_training_complete_false_when_missing(tmp_path: Path) -> None:
    assert not is_complete(tmp_path / "missing.json", expected_rounds=1)


def test_is_training_complete_false_when_short_of_rounds(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"server_evaluate_metrics": {"0": {}, "1": {}}}))
    assert not is_complete(path, expected_rounds=5)


def test_is_training_complete_false_on_unparseable_json(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("not json")
    assert not is_complete(path, expected_rounds=1)


def test_run_one_combo_skips_subprocess_when_already_complete(
    tmp_path: Path, monkeypatch
) -> None:
    name = _combo_name("homogeneous", "first-round", "vanilla", "fedavg")
    result_path = tmp_path / f"{name}.json"
    result_path.write_text(json.dumps({"server_evaluate_metrics": {"0": {}, "1": {}}}))
    (tmp_path / f"{name}.pt").write_bytes(b"fake-checkpoint")

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("subprocess.run should not be called when already complete")

    monkeypatch.setattr(subprocess, "run", fail_if_called)

    model_path, success = _run_training_combo(
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

    _model_path, success = _run_training_combo(
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

    _model_path, success = _run_training_combo(
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
        return 0.5, 0.6, 9

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

    failed_name = _combo_name("homogeneous", "first-round", "vanilla", "fedyogi")
    log_content = (tmp_path / "sweep_progress.log").read_text()
    assert failed_name in log_content
    assert "evaluation error" in log_content

    report_path = tmp_path / "cia_client_scaling.json"
    assert report_path.exists()
    report_data = json.loads(report_path.read_text())
    assert len(report_data["results"]) == 1
    assert report_data["results"][0]["aggregation"] == "fedavg"
    assert report_data["results"][0]["shadow_size"] == 9
    assert report_data["failed"] == [failed_name]


def test_run_one_combo_retrains_when_result_json_complete_but_checkpoint_missing(
    tmp_path: Path, monkeypatch
) -> None:
    """A crash between server.py writing the JSON and saving the .pt file must
    not be mistaken for a complete, resumable combo (Fix 2)."""
    name = _combo_name("homogeneous", "first-round", "vanilla", "fedavg")
    result_path = tmp_path / f"{name}.json"
    result_path.write_text(json.dumps({"server_evaluate_metrics": {"0": {}, "1": {}}}))
    # Deliberately no {name}.pt written alongside it.

    calls: list[list[str]] = []

    class FakeCompletedProcess:
        returncode = 0

    def fake_run(command, cwd=None):
        calls.append(command)
        return FakeCompletedProcess()

    monkeypatch.setattr(subprocess, "run", fake_run)

    model_path, success = _run_training_combo(
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
    assert len(calls) == 1, "training should be retried when the checkpoint is missing"


def test_run_cia_client_scaling_merges_across_separate_invocations(
    tmp_path: Path, monkeypatch
) -> None:
    """Running first-round then post-convergence separately must not clobber
    the first invocation's results (Fix 1)."""
    monkeypatch.setattr(
        subprocess, "run", lambda command, cwd=None: type("P", (), {"returncode": 0})()
    )
    monkeypatch.setattr(
        runner_module, "evaluate_combo", lambda model_path, *, partition_mode, device: (0.5, 0.6, 10)
    )

    runner_module.run_cia_client_scaling(
        output_dir=tmp_path,
        partition_modes=("homogeneous",),
        privacy_modes=("vanilla",),
        aggregations=("fedavg",),
        timings=("first-round",),
        max_parallel_clients=4,
        force=False,
    )
    runner_module.run_cia_client_scaling(
        output_dir=tmp_path,
        partition_modes=("homogeneous",),
        privacy_modes=("vanilla",),
        aggregations=("fedavg",),
        timings=("post-convergence",),
        max_parallel_clients=4,
        force=False,
    )

    report_data = json.loads((tmp_path / "cia_client_scaling.json").read_text())
    timings_present = {row["timing"] for row in report_data["results"]}
    assert timings_present == {"first-round", "post-convergence"}
    assert len(report_data["results"]) == 2


def test_run_cia_client_scaling_replaces_stale_entry_for_same_combo(
    tmp_path: Path, monkeypatch
) -> None:
    """Re-running a combo (e.g. with --force) should replace its old report
    row rather than duplicating it (Fix 1)."""
    monkeypatch.setattr(
        subprocess, "run", lambda command, cwd=None: type("P", (), {"returncode": 0})()
    )

    monkeypatch.setattr(
        runner_module, "evaluate_combo", lambda model_path, *, partition_mode, device: (1.0, 1.0, 10)
    )
    runner_module.run_cia_client_scaling(
        output_dir=tmp_path,
        partition_modes=("homogeneous",),
        privacy_modes=("vanilla",),
        aggregations=("fedavg",),
        timings=("first-round",),
        max_parallel_clients=4,
        force=False,
    )

    monkeypatch.setattr(
        runner_module, "evaluate_combo", lambda model_path, *, partition_mode, device: (0.2, 0.4, 12)
    )
    runner_module.run_cia_client_scaling(
        output_dir=tmp_path,
        partition_modes=("homogeneous",),
        privacy_modes=("vanilla",),
        aggregations=("fedavg",),
        timings=("first-round",),
        max_parallel_clients=4,
        force=True,
    )

    report_data = json.loads((tmp_path / "cia_client_scaling.json").read_text())
    assert len(report_data["results"]) == 1
    assert report_data["results"][0]["shadow_size"] == 12
    assert report_data["results"][0]["aggregated_test_loss"] == pytest.approx(0.2)


def test_main_rejects_invalid_timings_before_running_sweep(monkeypatch) -> None:
    """A typo in --timings must be caught instantly, before any subprocess
    call, instead of failing hours into a sweep (Fix 3)."""

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("run_cia_client_scaling should not be called")

    monkeypatch.setattr(runner_module, "run_cia_client_scaling", fail_if_called)
    monkeypatch.setattr(
        "sys.argv", ["prog", "--timings", "post-convergence,frist-round"]
    )

    with pytest.raises(SystemExit):
        main()


def test_resolve_noise_multiplier_defaults_to_timing_value() -> None:
    """Omitting the override must preserve the published per-timing defaults."""
    assert resolve_noise_multiplier("first-round", None) == 0.01
    assert resolve_noise_multiplier("post-convergence", None) == 0.05
    assert resolve_noise_multiplier("first-round", 0.12) == 0.12


def test_default_matrix_run_name_encodes_an_overridden_noise_multiplier() -> None:
    """Distinct noise levels must not collide on disk or skip each other."""
    name = _combo_name(
        "homogeneous", "first-round", "vanilla", "fedavg", noise_multiplier=0.12
    )
    assert "__nm0p12__" in name
    assert name != _combo_name("homogeneous", "first-round", "vanilla", "fedavg")


def test_build_combo_passes_overridden_noise(tmp_path) -> None:
    """The override must reach the underlying reproduce runner."""
    combo = build_combo(
        partition_mode="homogeneous",
        timing="first-round",
        privacy="metric-privacy",
        aggregation="fedyogi",
        noise_multiplier=0.12,
    )
    joined = " ".join(
        combo.runner_args(output_dir=tmp_path, max_parallel_clients=4, client_cpus=1.0)
    )
    assert "--noise-multiplier 0.12" in joined
    assert (
        "--run-name cia_scaling__first-round__homogeneous__metric-privacy__fedyogi__"
        "clients-48__seed-42__nm0p12__clip5__rounds-1__epochs-20" in joined
    )


def test_parse_subset_accepts_a_valid_subset() -> None:
    parser = _parser()
    assert parse_subset(parser, "fedyogi", AGGREGATIONS, "--aggregations") == ("fedyogi",)
    assert parse_subset(parser, "fedyogi,fedavg", AGGREGATIONS, "--aggregations") == (
        "fedyogi",
        "fedavg",
    )


def test_parse_subset_tolerates_whitespace_and_trailing_commas() -> None:
    parser = _parser()
    assert parse_subset(parser, " homogeneous , ", PARTITION_MODES, "--partitions") == (
        "homogeneous",
    )


def test_parse_subset_rejects_unknown_values() -> None:
    parser = _parser()
    with pytest.raises(SystemExit):
        parse_subset(parser, "fedsgd", AGGREGATIONS, "--aggregations")


def test_parse_subset_rejects_an_empty_selection() -> None:
    """An empty value must fail loudly rather than silently sweeping nothing."""
    parser = _parser()
    with pytest.raises(SystemExit):
        parse_subset(parser, " , ", PARTITION_MODES, "--partitions")


def test_cli_defaults_select_the_full_matrix() -> None:
    args = _parser().parse_args([])
    parser = _parser()
    assert parse_subset(parser, args.partitions, PARTITION_MODES, "-p") == PARTITION_MODES
    assert parse_subset(parser, args.aggregations, AGGREGATIONS, "-a") == AGGREGATIONS
    assert parse_subset(parser, args.privacy, PRIVACY_MODES, "-v") == PRIVACY_MODES
