from __future__ import annotations

import importlib
import json
import subprocess
import tarfile
from pathlib import Path

import pytest


@pytest.fixture
def setup_module(tmp_path, monkeypatch):
    monkeypatch.setenv("METRICDP_COLAB_CONTENT", str(tmp_path))
    from scripts.colab import remote_setup

    module = importlib.reload(remote_setup)
    source = tmp_path / "src"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname='x'\n")
    with tarfile.open(module.ARCHIVE, "w:gz") as archive:
        archive.add(source / "pyproject.toml", arcname="pyproject.toml")
    return module


def test_requirements_pin_lock_torch_and_exact_versions(setup_module):
    reqs = setup_module.requirements()
    assert "torch==2.10.0" in reqs and "torchvision==0.25.0" in reqs
    assert "flwr[simulation]==1.38.0" in reqs
    assert all("==" in r for r in reqs)


def test_mismatches_ignores_local_tag_and_reports_missing(setup_module):
    ok = dict(setup_module.PINNED) | {"torch": "2.10.0+cu128"}
    assert setup_module.mismatches(ok) == {}
    bad = dict(ok) | {"torch": "2.11.0+cu130"}
    bad.pop("ray")
    assert setup_module.mismatches(bad) == {
        "torch": ("2.10.0", "2.11.0+cu130"), "ray": ("2.55.1", None)
    }


def _fake_run(installed):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "nvidia-smi":
            return subprocess.CompletedProcess(cmd, 0, "580.82.07\n", "")
        if cmd[1:3] == ["-m", "pip"] and cmd[3] == "freeze":
            return subprocess.CompletedProcess(cmd, 0, "torch==2.10.0\n", "")
        if cmd[1] == "-c":
            info = dict(installed) | {"torch_cuda": "12.8", "cudnn": 91002,
                                      "cuda_available": True, "device": "A100"}
            return subprocess.CompletedProcess(cmd, 0, json.dumps(info) + "\n", "")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    return run, calls


def test_main_installs_pins_and_records_runtime(setup_module, monkeypatch):
    installed = dict(setup_module.PINNED) | {"torch": "2.10.0+cu128"}
    run, calls = _fake_run(installed)
    monkeypatch.setattr(setup_module.subprocess, "run", run)
    setup_module.main()
    install = calls[0]
    assert install[1:4] == ["-m", "pip", "install"] and "torch==2.10.0" in install
    runtime = json.loads(setup_module.RUNTIME_PATH.read_text())
    assert runtime["torch"] == "2.10.0+cu128" and runtime["cudnn"] == 91002
    assert runtime["driver"] == "580.82.07" and runtime["pinned"] == setup_module.PINNED
    assert setup_module.FREEZE_PATH.read_text() == "torch==2.10.0\n"


def test_main_fails_when_pin_not_honoured(setup_module, monkeypatch):
    installed = dict(setup_module.PINNED) | {"torch": "2.11.0+cu130"}
    run, _ = _fake_run(installed)
    monkeypatch.setattr(setup_module.subprocess, "run", run)
    with pytest.raises(SystemExit, match="torch"):
        setup_module.main()
    assert not setup_module.RUNTIME_PATH.exists()


def test_worker_records_runtime_and_freeze(tmp_path, monkeypatch):
    monkeypatch.setenv("METRICDP_COLAB_CONTENT", str(tmp_path))
    from scripts.colab import remote_worker

    worker = importlib.reload(remote_worker)
    (tmp_path / "metricdp-pytorch").mkdir()
    worker.RUNTIME_PATH.write_text(json.dumps({"torch": "2.10.0+cu128"}))
    worker.FREEZE_PATH.write_text("torch==2.10.0\n")
    worker.CONFIG_PATH.write_text(json.dumps({
        "results": "out", "module": "json.tool", "args": ["--help"],
        "source_commit": "abc", "source_branch": "b",
    }))
    worker.main()
    run = json.loads((tmp_path / "metricdp-pytorch/out/colab_run.json").read_text())
    assert run["runtime"] == {"torch": "2.10.0+cu128"} and run["state"] == "complete"
    assert (tmp_path / "metricdp-pytorch/out/colab_pip_freeze.txt").read_text() == "torch==2.10.0\n"
