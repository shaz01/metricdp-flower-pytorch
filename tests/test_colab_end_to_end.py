"""End-to-end test of the Colab controller against a fake ``colab`` CLI.

The real remote helpers (`remote_start.py`, `remote_worker.py`,
`remote_probe.py`) run for real here against a temporary content directory, so
this covers the parts that decide whether results survive: archive publication,
probe parsing, download, extraction, and the local commit. The fake CLI records
the ``HOME`` of every invocation, which is how per-account isolation is checked.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from scripts.colab import accounts as colab_accounts
from scripts.colab import run_experiment
from scripts.colab import state as colab_state

FAKE_COLAB = '''#!/usr/bin/env python3
"""Minimal stand-in for the google-colab-cli, backed by local directories."""

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(os.environ["FAKE_COLAB_ROOT"])
CALLS = ROOT / "calls.log"


def content(session):
    directory = ROOT / "sessions" / session
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def main():
    argv = sys.argv[1:]
    command = argv[0]
    CALLS.parent.mkdir(parents=True, exist_ok=True)
    with CALLS.open("a") as handle:
        handle.write(f"{os.environ.get('HOME')}\\t{' '.join(argv)}\\n")

    session = None
    if "-s" in argv:
        session = argv[argv.index("-s") + 1]
    rest = [value for value in argv[1:] if not value.startswith("-")]
    if session in rest:
        rest.remove(session)

    if command == "sessions":
        for directory in sorted((ROOT / "sessions").glob("*")):
            print(directory.name)
        return 0
    if command == "new":
        content(session)
        print(f"[colab] Session {session!r} READY.")
        return 0
    if command == "stop":
        shutil.rmtree(ROOT / "sessions" / session, ignore_errors=True)
        print(f"[colab] Stopped {session}")
        return 0
    if command == "upload":
        local, remote = rest[0], rest[1]
        shutil.copy2(local, content(session) / Path(remote).name)
        return 0
    if command == "download":
        remote, local = rest[0], rest[1]
        source = content(session) / Path(remote).name
        if not source.exists():
            print(f"missing {remote}", file=sys.stderr)
            return 1
        shutil.copy2(source, local)
        return 0
    if command == "exec":
        script = Path(argv[argv.index("-f") + 1])
        directory = content(session)
        if script.name == "remote_setup.py":
            # Stand in for the pip install: unpack the uploaded snapshot only.
            project = directory / "metricdp-pytorch"
            if project.exists():
                shutil.rmtree(project)
            project.mkdir(parents=True)
            with tarfile.open(directory / "metricdp-source.tar.gz") as archive:
                archive.extractall(project, filter="data")
            print(f"Colab environment ready at {project}")
            return 0
        environment = {**os.environ, "METRICDP_COLAB_CONTENT": str(directory)}
        result = subprocess.run(
            [sys.executable, str(script)], env=environment, capture_output=True, text=True
        )
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode
    print(f"unsupported command {command}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
'''

EXPERIMENT = '''"""Tiny stand-in experiment module."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()
    directory = Path(args.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    print("[ROUND 1/1] training")
    (directory / "run.json").write_text(json.dumps({"accuracy": 0.42}) + "\\n")
    if args.fail:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
'''


@pytest.fixture
def environment(tmp_path, monkeypatch):
    """A temporary repo, a fake ``colab`` on PATH, and one logged-in account."""
    repo = tmp_path / "repo"
    (repo / "fakeexp").mkdir(parents=True)
    (repo / "fakeexp" / "train.py").write_text(EXPERIMENT)
    (repo / "pyproject.toml").write_text('[project]\nname = "fake"\nversion = "0"\n')
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    for key, value in (("user.email", "t@example.com"), ("user.name", "Test")):
        subprocess.run(["git", "config", key, value], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)

    binary = tmp_path / "bin"
    binary.mkdir()
    fake = binary / "colab"
    fake.write_text(FAKE_COLAB)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("PATH", f"{binary}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_COLAB_ROOT", str(tmp_path / "colab"))
    monkeypatch.setenv("METRICDP_COLAB_ACCOUNTS_DIR", str(tmp_path / "accounts"))
    colab_accounts.prepare_account("lab2")
    colab_accounts.token_path("lab2").write_text(
        json.dumps({"refresh_token": "r", "account": "lab2@example.com"})
    )

    monkeypatch.setattr(run_experiment, "SOURCE_ROOTS", ("fakeexp",))
    monkeypatch.setattr(run_experiment, "PROJECT_ROOT", repo)
    monkeypatch.setattr(colab_state, "PROJECT_ROOT", repo)
    monkeypatch.setattr(colab_state, "STATE_DIR", repo / ".colab")
    monkeypatch.setattr(colab_state, "LOG_DIR", repo / ".colab" / "logs")
    monkeypatch.setattr(colab_state, "STAGING_DIR", repo / ".colab" / "staging")
    monkeypatch.setattr(colab_state, "COMMIT_LOCK_PATH", repo / ".colab" / "commit.lock")
    monkeypatch.setattr(run_experiment, "_notify_macos", lambda *args: None)
    return repo


def _run_args(session: str, results: str, extra: list[str] | None = None):
    return argparse.Namespace(
        session=session,
        gpu="A100",
        module="fakeexp.train",
        results=results,
        commit_message=f"results(fake): {session}",
        account="auto",
        max_per_account=2,
        force=False,
        attach=True,
        poll_seconds=1,
        module_args=["--", "--output-dir", results, *(extra or [])],
    )


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def test_run_collects_commits_and_releases_without_pushing(environment) -> None:
    repo = environment
    run_experiment.run_job(_run_args("fake-a", "results/fake/a"))

    result = repo / "results" / "fake" / "a"
    assert json.loads((result / "run.json").read_text())["accuracy"] == 0.42
    assert (result / "colab_run.json").exists()
    assert (result / "colab_training.log").exists()

    state = colab_state.load_state("fake-a")
    assert colab_state.phase(state) == colab_state.PHASE_COLLECTED
    assert state["remote_state"] == "complete"
    assert state["commit"]

    assert "results(fake): fake-a" in _git(repo, "log", "--oneline", "-1")
    assert _git(repo, "status", "--porcelain", "--", "results") == ""
    # No remote is configured, so a surviving push step would have failed.
    assert _git(repo, "remote") == ""
    # The VM was released.
    assert not (Path(os.environ["FAKE_COLAB_ROOT"]) / "sessions" / "fake-a").exists()


def test_every_remote_call_uses_the_selected_account_home(environment) -> None:
    run_experiment.run_job(_run_args("fake-b", "results/fake/b"))

    calls = (Path(os.environ["FAKE_COLAB_ROOT"]) / "calls.log").read_text().splitlines()
    homes = {line.split("\t")[0] for line in calls}

    assert homes == {str(colab_accounts.account_home("lab2"))}
    assert colab_state.load_state("fake-b")["account"] == "lab2"
    assert any("new -s fake-b" in line for line in calls)
    assert any("stop -s fake-b" in line for line in calls)


def test_failed_experiment_still_commits_its_artifacts(environment) -> None:
    repo = environment
    with pytest.raises(run_experiment.ExperimentFailed):
        run_experiment.run_job(_run_args("fake-c", "results/fake/c", ["--fail"]))

    state = colab_state.load_state("fake-c")
    assert state["remote_state"] == "failed"
    assert colab_state.phase(state) == colab_state.PHASE_COLLECTED
    assert "(failed run artifacts)" in _git(repo, "log", "--oneline", "-1")
    assert (repo / "results" / "fake" / "c" / "colab_run.json").exists()


def test_concurrent_sessions_commit_disjoint_directories(environment) -> None:
    repo = environment
    run_experiment.run_job(_run_args("fake-d", "results/fake/d"))
    run_experiment.run_job(_run_args("fake-e", "results/fake/e"))

    subjects = _git(repo, "log", "--format=%s", "-2").splitlines()
    assert subjects == ["results(fake): fake-e", "results(fake): fake-d"]
    first = _git(repo, "show", "--name-only", "--format=", "HEAD~1").split()
    assert all(path.startswith("results/fake/d/") for path in first)
    second = _git(repo, "show", "--name-only", "--format=", "HEAD").split()
    assert all(path.startswith("results/fake/e/") for path in second)


def test_sweep_collects_a_run_whose_controller_disappeared(environment) -> None:
    repo = environment
    state = {
        "session": "fake-f",
        "account": "lab2",
        "gpu": "A100",
        "module": "fakeexp.train",
        "args": ["--output-dir", "results/fake/f"],
        "results": "results/fake/f",
        "commit_message": "results(fake): fake-f",
        "source_branch": _git(repo, "branch", "--show-current"),
        "source_commit": _git(repo, "rev-parse", "HEAD"),
        "phase": colab_state.PHASE_LAUNCHING,
        "launched_at": colab_state.now(),
        "poll_seconds": 1,
        "controller_pid": None,
    }
    colab_state.write_state(state)
    run_experiment._stage_source(state)
    run_experiment._provision(state)
    run_experiment._upload_and_start(state)
    colab_state.update_state("fake-f", phase=colab_state.PHASE_TRAINING)
    # The controller is gone; nothing would ever collect this run on its own.

    for _ in range(30):
        if (
            run_experiment._probe("fake-f", "lab2", echo=False)["state"] == "complete"
        ):
            break
    else:
        pytest.fail("remote worker never reported completion")

    run_experiment.sweep(argparse.Namespace(session=None, dry_run=False))

    assert (repo / "results" / "fake" / "f" / "run.json").exists()
    assert colab_state.phase(colab_state.load_state("fake-f")) == (
        colab_state.PHASE_COLLECTED
    )
    assert "results(fake): fake-f" in _git(repo, "log", "--oneline", "-1")


def test_results_archive_is_only_published_when_complete(environment) -> None:
    """A probe reporting ``complete`` implies a fully written archive."""
    repo = environment
    state = {
        "session": "fake-g",
        "account": "lab2",
        "gpu": "A100",
        "module": "fakeexp.train",
        "args": ["--output-dir", "results/fake/g"],
        "results": "results/fake/g",
        "commit_message": "results(fake): fake-g",
        "source_branch": _git(repo, "branch", "--show-current"),
        "source_commit": _git(repo, "rev-parse", "HEAD"),
        "phase": colab_state.PHASE_LAUNCHING,
        "launched_at": colab_state.now(),
    }
    colab_state.write_state(state)
    run_experiment._stage_source(state)
    run_experiment._provision(state)
    run_experiment._upload_and_start(state)

    seen = []
    for _ in range(60):
        probe = run_experiment._probe("fake-g", "lab2", echo=False)
        seen.append(probe)
        if probe["state"] == "complete":
            break
    else:
        pytest.fail("remote worker never reported completion")

    for probe in seen:
        archive = probe["raw"].split("--- results archive ---")[1].split()[0]
        if probe["state"] == "complete":
            assert int(archive) > 0
            assert probe["status"]["archive_bytes"] == int(archive)
