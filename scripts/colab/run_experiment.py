"""Run a repository experiment on Colab and push its results from local Git."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HELPER_DIR = Path(__file__).resolve().parent
STATE_DIR = PROJECT_ROOT / ".colab"
SOURCE_ROOTS = ("metricdp_pytorch", "experiments", "scripts")
SOURCE_FILES = ("pyproject.toml", "uv.lock", ".python-version", "README.md", "LICENSE")
SECRET_PATTERNS = (
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)


def _run(
    *args: str,
    check: bool = True,
    capture: bool = False,
    local_timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        check=check,
        text=True,
        capture_output=capture,
        timeout=local_timeout,
    )


def _git(*args: str) -> str:
    return _run("git", *args, capture=True).stdout.strip()


def _state_path(session: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", session):
        raise ValueError("session must contain only letters, digits, '_' or '-'")
    return STATE_DIR / f"{session}.json"


def _load_state(session: str) -> dict[str, Any]:
    path = _state_path(session)
    if not path.exists():
        raise FileNotFoundError(f"No local state for Colab session {session!r}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(state: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(state["session"]).write_text(
        json.dumps(state, indent=2) + "\n", encoding="utf-8"
    )


def _validate_results_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("results",):
        raise ValueError("--results must be a relative path below results/")
    return path


def _forwarded_module_args(values: list[str]) -> list[str]:
    """Remove argparse's delimiter before forwarding experiment arguments."""
    return values[1:] if values[:1] == ["--"] else values


def _source_paths() -> list[Path]:
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
        "--",
        *SOURCE_ROOTS,
        *SOURCE_FILES,
    ]
    output = subprocess.run(
        command, cwd=PROJECT_ROOT, check=True, capture_output=True
    ).stdout
    paths = []
    for raw_path in output.split(b"\0"):
        if not raw_path:
            continue
        path = Path(raw_path.decode())
        if path.suffix == ".ipynb" or not (PROJECT_ROOT / path).is_file():
            continue
        paths.append(path)
    return sorted(set(paths))


def _make_source_archive(destination: Path) -> None:
    paths = _source_paths()
    for path in paths:
        data = (PROJECT_ROOT / path).read_bytes()
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise RuntimeError(f"Refusing to upload source containing a credential: {path}")
    with tarfile.open(destination, "w:gz") as archive:
        for path in paths:
            archive.add(PROJECT_ROOT / path, arcname=path)


def _colab(
    session: str,
    command: str,
    *args: str,
    timeout: str | None = None,
    local_timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    invocation = ["colab", command, "-s", session]
    if timeout is not None:
        invocation.extend(("--timeout", timeout))
    invocation.extend(args)
    try:
        return _run(*invocation, capture=True, local_timeout=local_timeout)
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, end="")
        if error.stderr:
            print(error.stderr, end="")
        raise


def _provision(state: dict[str, Any]) -> None:
    session = state["session"]
    print(f"Provisioning Colab session {session!r} with {state['gpu']} GPU...")
    _run("colab", "new", "-s", session, "--gpu", state["gpu"])


def _upload_and_start(state: dict[str, Any], archive: Path, config_path: Path) -> None:
    session = state["session"]
    uploads = (
        (archive, "/content/metricdp-source.tar.gz"),
        (config_path, "/content/metricdp-colab-job.json"),
        (HELPER_DIR / "remote_worker.py", "/content/metricdp-colab-worker.py"),
    )
    for local, remote in uploads:
        _colab(session, "upload", str(local), remote)
    setup = _colab(
        session,
        "exec",
        "-f",
        str(HELPER_DIR / "remote_setup.py"),
        timeout="1800",
    )
    print(setup.stdout, end="")
    started = _colab(
        session,
        "exec",
        "-f",
        str(HELPER_DIR / "remote_start.py"),
        timeout="120",
    )
    print(started.stdout, end="")


def _probe(session: str) -> tuple[str, str]:
    try:
        result = _colab(
            session,
            "exec",
            "-f",
            str(HELPER_DIR / "remote_probe.py"),
            timeout="30",
            local_timeout=45,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        print(f"Colab status probe failed ({error}); retrying without stopping training.")
        return "unknown", ""
    output = result.stdout
    print(output, end="")
    marker = "COLAB_JOB_STATUS="
    start = output.find(marker)
    if start < 0:
        return "unknown", output
    decoder = json.JSONDecoder()
    status, _ = decoder.raw_decode(output[start + len(marker) :].lstrip())
    return str(status.get("state", "unknown")), output


def _extract_results(archive_path: Path, expected: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="metricdp-colab-results-") as temp_name:
        temp = Path(temp_name)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(temp, filter="data")
        source = temp / expected
        if not source.is_dir():
            raise RuntimeError(f"Downloaded archive does not contain {expected}")
        destination = PROJECT_ROOT / expected
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, dirs_exist_ok=True)


def _commit_and_push(state: dict[str, Any], remote_state: str) -> None:
    branch = _git("branch", "--show-current")
    if branch != state["source_branch"]:
        raise RuntimeError(
            f"Current branch changed from {state['source_branch']!r} to {branch!r}; "
            "results were downloaded but were not committed to the wrong branch."
        )
    results = state["results"]
    _run("git", "add", "--", results, capture=True)
    changed = (
        _run("git", "diff", "--cached", "--quiet", "--", results, check=False).returncode
        != 0
    )
    if changed:
        message = state["commit_message"]
        if remote_state != "complete":
            message += " (failed run artifacts)"
        _run("git", "commit", "-m", message, "--", results)
    _run("git", "push", "origin", f"HEAD:refs/heads/{branch}")
    print(f"Results committed and pushed to origin/{branch}")


def collect(session: str, *, stop: bool = True) -> str:
    state = _load_state(session)
    remote_state, _ = _probe(session)
    if remote_state not in {"complete", "failed"}:
        raise RuntimeError(f"Colab job is not finished (state={remote_state})")
    with tempfile.TemporaryDirectory(prefix="metricdp-colab-download-") as temp_name:
        archive_path = Path(temp_name) / "results.tar.gz"
        _colab(
            session,
            "download",
            "/content/metricdp-colab-results.tar.gz",
            str(archive_path),
        )
        _extract_results(archive_path, Path(state["results"]))
    _commit_and_push(state, remote_state)
    if stop:
        _colab(session, "stop")
        print(f"Released Colab session {session!r}")
    return remote_state


def run_job(args: argparse.Namespace) -> None:
    results = _validate_results_path(args.results)
    branch = _git("branch", "--show-current")
    if not branch:
        raise RuntimeError("Colab runs require a named Git branch")
    state = {
        "session": args.session,
        "gpu": args.gpu,
        "module": args.module,
        "args": _forwarded_module_args(args.module_args),
        "results": results.as_posix(),
        "commit_message": args.commit_message,
        "source_branch": branch,
        "source_commit": _git("rev-parse", "HEAD"),
    }
    _save_state(state)

    provisioned = False
    try:
        with tempfile.TemporaryDirectory(prefix="metricdp-colab-source-") as temp_name:
            temp = Path(temp_name)
            archive = temp / "source.tar.gz"
            config_path = temp / "job.json"
            _make_source_archive(archive)
            config_path.write_text(json.dumps(state) + "\n", encoding="utf-8")
            _provision(state)
            provisioned = True
            _upload_and_start(state, archive, config_path)
        relative_script = Path(__file__).relative_to(PROJECT_ROOT)
        print(
            f"Training started. Live checks: uv run python {relative_script} "
            f"status --session {args.session}"
        )
        while True:
            remote_state, _ = _probe(args.session)
            if remote_state in {"complete", "failed"}:
                break
            time.sleep(args.poll_seconds)
        final_state = collect(args.session)
        if final_state != "complete":
            raise RuntimeError("Colab experiment failed; partial artifacts were pushed")
    except KeyboardInterrupt:
        relative_script = Path(__file__).relative_to(PROJECT_ROOT)
        print(
            "\nLocal monitoring stopped; the Colab job is still running. "
            f"Resume with: uv run python {relative_script} wait --session {args.session}"
        )
        raise
    except Exception:
        if provisioned:
            relative_script = Path(__file__).relative_to(PROJECT_ROOT)
            print(
                f"Session {args.session!r} was preserved for recovery. Inspect it with "
                f"`uv run python {relative_script} status --session {args.session}`."
            )
        raise


def wait_for_job(session: str, poll_seconds: int) -> None:
    while True:
        remote_state, _ = _probe(session)
        if remote_state in {"complete", "failed"}:
            break
        time.sleep(poll_seconds)
    final_state = collect(session)
    if final_state != "complete":
        raise RuntimeError("Colab experiment failed; partial artifacts were pushed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="provision, run, collect, push, stop")
    run_parser.add_argument("--session", required=True)
    run_parser.add_argument(
        "--gpu", default="L4", choices=("T4", "L4", "G4", "A100", "H100")
    )
    run_parser.add_argument("--module", required=True)
    run_parser.add_argument("--results", required=True)
    run_parser.add_argument("--commit-message", required=True)
    run_parser.add_argument("--poll-seconds", type=int, default=60)
    run_parser.add_argument("module_args", nargs=argparse.REMAINDER)

    for name in ("status", "collect", "wait", "stop"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--session", required=True)
        if name == "wait":
            subparser.add_argument("--poll-seconds", type=int, default=60)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run_job(args)
    elif args.command == "status":
        _probe(args.session)
    elif args.command == "collect":
        collect(args.session)
    elif args.command == "wait":
        wait_for_job(args.session, args.poll_seconds)
    elif args.command == "stop":
        _colab(args.session, "stop")


if __name__ == "__main__":
    main()
