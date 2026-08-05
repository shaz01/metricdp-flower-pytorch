"""Tests for local Colab controller argument handling and notifications."""

from scripts.colab import run_experiment
from scripts.colab.run_experiment import _forwarded_module_args


def test_module_argument_delimiter_is_not_forwarded() -> None:
    assert _forwarded_module_args(["--", "--suite", "fashion"]) == [
        "--suite",
        "fashion",
    ]


def test_module_arguments_without_delimiter_are_unchanged() -> None:
    assert _forwarded_module_args(["value"]) == ["value"]


def test_macos_notification_uses_osascript(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(run_experiment.sys, "platform", "darwin")
    monkeypatch.setattr(
        run_experiment.shutil, "which", lambda _: "/usr/bin/osascript"
    )
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
