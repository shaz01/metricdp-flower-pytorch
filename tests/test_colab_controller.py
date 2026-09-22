"""Tests for the multi-account Colab controller: accounts, state, and sweep logic."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from scripts.colab import accounts as colab_accounts
from scripts.colab import run_experiment
from scripts.colab import state as colab_state
from scripts.colab.run_experiment import _forwarded_module_args


@pytest.fixture
def state_dir(tmp_path, monkeypatch):
    """Redirect every state file, log, and lock into a temporary directory."""
    directory = tmp_path / ".colab"
    monkeypatch.setattr(colab_state, "STATE_DIR", directory)
    monkeypatch.setattr(colab_state, "LOG_DIR", directory / "logs")
    monkeypatch.setattr(colab_state, "STAGING_DIR", directory / "staging")
    monkeypatch.setattr(colab_state, "COMMIT_LOCK_PATH", directory / "commit.lock")
    return directory


def _state(session: str, **fields) -> dict:
    base = {
        "session": session,
        "account": "default",
        "phase": colab_state.PHASE_TRAINING,
        "launched_at": colab_state.now(),
        "results": f"results/{session}",
    }
    base.update(fields)
    return colab_state.write_state(base)


# --------------------------------------------------------------------------- #
# Argument handling and notifications
# --------------------------------------------------------------------------- #


def test_module_argument_delimiter_is_not_forwarded() -> None:
    assert _forwarded_module_args(["--", "--suite", "fashion"]) == [
        "--suite",
        "fashion",
    ]


def test_module_arguments_without_delimiter_are_unchanged() -> None:
    assert _forwarded_module_args(["value"]) == ["value"]


def test_detached_waiter_does_not_recursively_detach() -> None:
    command = run_experiment._controller_command("wait", "session", 240)
    assert command[-1] == "--attach"
    assert "--attach" not in run_experiment._controller_command("launch", "session", 240)


def test_results_path_must_stay_under_results() -> None:
    assert run_experiment._validate_results_path("results/cia/run").parts[0] == "results"
    for bad in ("/tmp/run", "results/../etc", "reports/run"):
        with pytest.raises(ValueError):
            run_experiment._validate_results_path(bad)


def test_output_dir_must_match_the_collected_directory() -> None:
    results = "results/cia/run"
    run_experiment._check_output_dir(["--output-dir", results], results)
    run_experiment._check_output_dir([f"--output-dir={results}"], results)
    run_experiment._check_output_dir(["--clients", "8"], results)

    with pytest.raises(RuntimeError, match="only the --results directory"):
        run_experiment._check_output_dir(["--output-dir", "results/other"], results)


def test_macos_notification_uses_osascript(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(run_experiment.sys, "platform", "darwin")
    monkeypatch.setattr(run_experiment.shutil, "which", lambda _: "/usr/bin/osascript")
    monkeypatch.setattr(
        run_experiment.subprocess,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    run_experiment._notify_macos("Collected", "Session complete")

    assert calls[0][0][0][:2] == ["osascript", "-e"]
    assert calls[0][0][0][-2:] == ["Collected", "Session complete"]


def test_notification_is_skipped_off_macos(monkeypatch) -> None:
    def unexpected_run(*_args, **_kwargs) -> None:
        raise AssertionError("osascript must not run off macOS")

    monkeypatch.setattr(run_experiment.sys, "platform", "linux")
    monkeypatch.setattr(run_experiment.subprocess, "run", unexpected_run)

    run_experiment._notify_macos("Collected", "Session complete")


# --------------------------------------------------------------------------- #
# Account isolation
# --------------------------------------------------------------------------- #


def test_each_account_gets_its_own_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("METRICDP_COLAB_ACCOUNTS_DIR", str(tmp_path / "accounts"))

    assert colab_accounts.account_home("default") == Path.home()
    assert colab_accounts.account_home("lab2") == tmp_path / "accounts" / "lab2"
    assert colab_accounts.token_path("lab2").name == "token.json"

    environment = colab_accounts.account_env("lab2")
    assert environment["HOME"] == str(tmp_path / "accounts" / "lab2")
    assert os.environ["HOME"] != environment["HOME"]


def test_invalid_account_names_are_rejected() -> None:
    for bad in ("", "../escape", "has space", "-leading"):
        with pytest.raises(ValueError):
            colab_accounts.validate_account(bad)


def test_accounts_are_discovered_and_login_needs_a_refresh_token(
    tmp_path, monkeypatch
) -> None:
    root = tmp_path / "accounts"
    monkeypatch.setenv("METRICDP_COLAB_ACCOUNTS_DIR", str(root))
    colab_accounts.prepare_account("lab2")
    colab_accounts.prepare_account("lab3")
    colab_accounts.token_path("lab2").write_text(
        json.dumps({"refresh_token": "r", "account": "lab2@example.com"})
    )
    colab_accounts.token_path("lab3").write_text(json.dumps({"token": "no-refresh"}))

    assert colab_accounts.known_accounts() == ["default", "lab2", "lab3"]
    assert colab_accounts.is_logged_in("lab2")
    assert not colab_accounts.is_logged_in("lab3")
    assert colab_accounts.account_email("lab2") == "lab2@example.com"
    assert "lab2" in colab_accounts.logged_in_accounts()
    assert "lab3" not in colab_accounts.logged_in_accounts()


# --------------------------------------------------------------------------- #
# Slot selection
# --------------------------------------------------------------------------- #


def _fake_accounts(monkeypatch, names: list[str]) -> None:
    monkeypatch.setattr(colab_accounts, "known_accounts", lambda: names)
    monkeypatch.setattr(colab_accounts, "logged_in_accounts", lambda: names)
    monkeypatch.setattr(colab_accounts, "is_logged_in", lambda account: account in names)


def test_auto_account_picks_the_least_loaded_account(state_dir, monkeypatch) -> None:
    _fake_accounts(monkeypatch, ["default", "lab2"])
    _state("a", account="default")
    _state("b", account="default", phase=colab_state.PHASE_LAUNCHING)

    assert run_experiment._select_account("auto", 2, False) == "lab2"


def test_auto_account_fails_when_every_account_is_capped(state_dir, monkeypatch) -> None:
    _fake_accounts(monkeypatch, ["default", "lab2"])
    for index, account in enumerate(("default", "default", "lab2", "lab2")):
        _state(f"s{index}", account=account)

    with pytest.raises(RuntimeError, match="2-session cap"):
        run_experiment._select_account("auto", 2, False)


def test_collected_sessions_free_their_slot(state_dir, monkeypatch) -> None:
    _fake_accounts(monkeypatch, ["default"])
    _state("done", account="default", phase=colab_state.PHASE_COLLECTED)
    _state("stopped", account="default", phase=colab_state.PHASE_STOPPED)

    assert run_experiment._select_account("auto", 2, False) == "default"


def test_explicit_account_respects_the_cap_unless_forced(state_dir, monkeypatch) -> None:
    _fake_accounts(monkeypatch, ["default", "lab2"])
    _state("a", account="lab2")
    _state("b", account="lab2")

    with pytest.raises(RuntimeError, match="already has 2 active"):
        run_experiment._select_account("lab2", 2, False)
    assert run_experiment._select_account("lab2", 2, True) == "lab2"


def test_explicit_account_must_be_logged_in(state_dir, monkeypatch) -> None:
    _fake_accounts(monkeypatch, ["default"])

    with pytest.raises(RuntimeError, match="not logged in"):
        run_experiment._select_account("lab9", 2, False)


# --------------------------------------------------------------------------- #
# State files and the commit lock
# --------------------------------------------------------------------------- #


def test_update_state_merges_instead_of_replacing(state_dir) -> None:
    _state("merge", account="lab2")
    colab_state.update_state("merge", controller_pid=42)
    colab_state.update_state("merge", phase=colab_state.PHASE_COLLECTED)

    stored = colab_state.load_state("merge")
    assert stored["controller_pid"] == 42
    assert stored["account"] == "lab2"
    assert stored["phase"] == colab_state.PHASE_COLLECTED


def test_pre_phase_state_files_count_as_finished(state_dir) -> None:
    colab_state.write_state({"session": "legacy", "results": "results/legacy"})
    assert colab_state.phase(colab_state.load_state("legacy")) == (
        colab_state.PHASE_COLLECTED
    )
    assert colab_state.active_states() == []


def test_commit_lock_is_exclusive(state_dir) -> None:
    with colab_state.commit_lock():
        with pytest.raises(TimeoutError):
            with colab_state.commit_lock(timeout=0.0):
                raise AssertionError("second holder must not enter the commit section")


def test_commit_lock_is_reacquirable_after_release(state_dir) -> None:
    with colab_state.commit_lock():
        pass
    with colab_state.commit_lock(timeout=1.0):
        pass


def test_controller_liveness_requires_a_matching_process(state_dir) -> None:
    assert not colab_state.controller_alive(_state("dead", controller_pid=None))
    assert not colab_state.controller_alive(_state("gone", controller_pid=2**21))
    # This test process is alive but is not a controller for the session.
    assert not colab_state.controller_alive(_state("mine", controller_pid=os.getpid()))


# --------------------------------------------------------------------------- #
# Probe parsing
# --------------------------------------------------------------------------- #

PROBE_OUTPUT = """COLAB_JOB_STATUS={"state": "running", "pid": 1234}

--- recent training output ---
INFO :      [ROUND 14/20]
INFO :      aggregated fit

--- nvidia-smi ---
2026/08/14 03:17:08.996, NVIDIA A100-SXM4-40GB, 47 %, 6408 MiB, 40960 MiB
"""


def test_probe_output_is_split_into_status_log_and_gpu() -> None:
    probe = run_experiment._parse_probe(PROBE_OUTPUT)

    assert probe["state"] == "running"
    assert probe["status"]["pid"] == 1234
    assert "ROUND 14/20" in probe["log_tail"]
    assert "nvidia-smi" not in probe["log_tail"]
    assert "47 %" in probe["gpu"]


def test_progress_reads_round_and_gpu_utilization() -> None:
    round_text, gpu_text, last_line = run_experiment._progress(
        run_experiment._parse_probe(PROBE_OUTPUT)
    )

    assert round_text == "14/20"
    assert gpu_text == "47%"
    assert last_line == "INFO :      aggregated fit"


def test_probe_without_status_marker_is_unknown() -> None:
    assert run_experiment._parse_probe("kernel is busy")["state"] == "unknown"


def test_packaging_is_not_a_terminal_remote_state() -> None:
    """The worker publishes its archive before announcing completion."""
    probe = run_experiment._parse_probe('COLAB_JOB_STATUS={"state": "packaging"}')
    assert probe["state"] not in {"complete", "failed"}


# --------------------------------------------------------------------------- #
# Sweep decisions
# --------------------------------------------------------------------------- #


def _decide(state: dict, *, remote: str, alive: bool, phase: str | None = None):
    row = {
        "phase": phase or colab_state.phase(state),
        "remote": remote,
        "alive": alive,
        "last_line": "",
    }
    return run_experiment._decide(state, row)


def test_finished_run_without_a_controller_is_collected(state_dir) -> None:
    state = _state("done")
    action, task = _decide(state, remote="complete", alive=False)

    assert task == "collect"
    assert not action.startswith("!")


def test_finished_run_with_a_live_controller_is_left_alone(state_dir) -> None:
    state = _state("busy")
    action, task = _decide(state, remote="complete", alive=True)

    assert task is None
    assert "controller collecting" in action


def test_failed_run_is_still_collected_for_its_artifacts(state_dir) -> None:
    state = _state("broken")
    _action, task = _decide(state, remote="failed", alive=False)

    assert task == "collect"


def test_running_job_whose_controller_died_gets_a_new_waiter(state_dir) -> None:
    state = _state("orphan")
    action, task = _decide(state, remote="running", alive=False)

    assert task == "adopt"
    assert action.startswith("!")


def test_healthy_run_needs_no_action(state_dir) -> None:
    state = _state("healthy", last_progress_at=colab_state.now())
    action, task = _decide(state, remote="running", alive=True)

    assert (action, task) == ("-", None)


def test_stalled_log_is_flagged(state_dir) -> None:
    state = _state("stalled", last_progress_at="2000-01-01T00:00:00+00:00")
    action, task = _decide(state, remote="running", alive=True)

    assert task is None
    assert action.startswith("!") and "no log progress" in action


def test_launch_that_never_starts_training_is_flagged(state_dir) -> None:
    state = _state(
        "hung",
        phase=colab_state.PHASE_LAUNCHING,
        launched_at="2000-01-01T00:00:00+00:00",
    )
    action, task = _decide(state, remote="not-started", alive=True)

    assert task is None
    assert action.startswith("!") and "no training after" in action


def test_recent_launch_is_not_flagged(state_dir) -> None:
    state = _state("young", phase=colab_state.PHASE_LAUNCHING)
    action, task = _decide(state, remote="not-started", alive=True)

    assert (action, task) == ("starting", None)


def test_unreachable_vm_during_launch_is_not_reported_as_a_probe_failure(
    state_dir,
) -> None:
    state = _state("provisioning", phase=colab_state.PHASE_LAUNCHING)
    action, task = _decide(state, remote="unknown", alive=True)

    assert (action, task) == ("starting", None)


def test_recycled_vm_is_flagged_rather_than_adopted(state_dir) -> None:
    """A worker that vanished mid-training means Colab reset the VM."""
    state = _state("recycled")
    action, task = _decide(state, remote="not-started", alive=True)

    assert task is None
    assert action.startswith("!") and "VM was reset" in action


def test_probe_failure_is_reported_without_acting(state_dir) -> None:
    state = _state("unreachable")
    _action, task = _decide(state, remote="unknown", alive=False)

    assert task is None


def test_sweep_collects_orphans_adopts_runners_and_reports(
    state_dir, monkeypatch, capsys
) -> None:
    _state("finished", account="lab2")
    _state("running", account="default")
    _state("mine", account="default", controller_pid=os.getpid())

    probes = {
        "finished": run_experiment._parse_probe('COLAB_JOB_STATUS={"state":"complete"}'),
        "running": run_experiment._parse_probe(PROBE_OUTPUT),
        "mine": run_experiment._parse_probe(PROBE_OUTPUT),
    }
    monkeypatch.setattr(
        run_experiment, "_probe", lambda session, account, echo=True: probes[session]
    )
    monkeypatch.setattr(
        colab_state,
        "controller_alive",
        lambda state: state["session"] == "mine",
    )
    collected: list[str] = []
    monkeypatch.setattr(
        run_experiment,
        "collect",
        lambda session, remote_state=None: collected.append(session) or "complete",
    )
    adopted: list[list[str]] = []
    monkeypatch.setattr(
        run_experiment, "_detach", lambda session, command: adopted.append(command) or 99
    )

    run_experiment.sweep(
        argparse.Namespace(session=None, dry_run=False)
    )
    output = capsys.readouterr().out

    assert collected == ["finished"]
    assert adopted and adopted[0][2] == "wait"
    assert "SESSION" in output and "need attention" in output
    # The healthy session with a live controller is reported and left alone.
    assert "14/20" in output and "47%" in output
    assert colab_state.load_state("mine")["last_log_line"].endswith("aggregated fit")


def test_sweep_dry_run_acts_on_nothing(state_dir, monkeypatch, capsys) -> None:
    _state("finished")
    monkeypatch.setattr(
        run_experiment,
        "_probe",
        lambda session, account, echo=True: run_experiment._parse_probe(
            'COLAB_JOB_STATUS={"state":"complete"}'
        ),
    )
    monkeypatch.setattr(colab_state, "controller_alive", lambda state: False)
    monkeypatch.setattr(
        run_experiment,
        "collect",
        lambda *args, **kwargs: pytest.fail("dry run must not collect"),
    )

    run_experiment.sweep(argparse.Namespace(session=None, dry_run=True))

    assert "complete" in capsys.readouterr().out
    assert colab_state.phase(colab_state.load_state("finished")) == (
        colab_state.PHASE_TRAINING
    )


def test_errored_session_is_surfaced_not_retried(state_dir) -> None:
    state = _state("bad", phase=colab_state.PHASE_ERROR, last_error="RuntimeError: boom")
    action, task = _decide(state, remote="running", alive=False)

    assert task is None
    assert action.startswith("!") and "boom" in action
