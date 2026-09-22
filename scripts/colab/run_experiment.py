"""Run repository experiments on several Colab accounts and commit their results.

Controllers detach by default, so a launch returns immediately and many GPUs can
run at once across accounts. ``sweep`` is the single observation command: it
probes every recorded session, auto-collects finished work whose controller is
gone, and flags what needs a human. Results are committed locally under a shared
lock; pushing stays manual.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.colab import accounts as colab_accounts  # noqa: E402
from scripts.colab import state as colab_state  # noqa: E402

HELPER_DIR = Path(__file__).resolve().parent
SOURCE_ROOTS = ("metricdp_pytorch", "experiments", "scripts")
SOURCE_FILES = ("pyproject.toml", "uv.lock", ".python-version", "README.md", "LICENSE")
SECRET_PATTERNS = (
    re.compile(rb"ghp_[A-Za-z0-9]{20,}"),
    re.compile(rb"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
)
DEFAULT_SESSIONS_PER_ACCOUNT = 2
LAUNCH_WATCHDOG_MINUTES = 10.0
STALL_WATCHDOG_MINUTES = 15.0
ROUND_PATTERN = re.compile(r"\[ROUND (\d+)\s*/\s*(\d+)\]")
GPU_PATTERN = re.compile(r",\s*(\d+)\s*%,\s*(\d+)\s*MiB")


class ExperimentFailed(RuntimeError):
    """The remote module exited non-zero. Its artifacts were still collected."""


def _notify_macos(title: str, message: str) -> None:
    """Send a best-effort macOS notification without affecting the controller."""
    if sys.platform != "darwin" or shutil.which("osascript") is None:
        return
    script = (
        "on run argv\n"
        " display notification (item 2 of argv) with title (item 1 of argv)\n"
        "end run"
    )
    try:
        subprocess.run(
            ["osascript", "-e", script, title, message],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass


def _run(
    *args: str,
    check: bool = True,
    capture: bool = False,
    local_timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        check=check,
        text=True,
        capture_output=capture,
        timeout=local_timeout,
        env=env,
    )


def _git(*args: str) -> str:
    return _run("git", *args, capture=True).stdout.strip()


def _forwarded_module_args(values: list[str]) -> list[str]:
    """Remove argparse's delimiter before forwarding experiment arguments."""
    return values[1:] if values[:1] == ["--"] else values


def _display_path(path: Path) -> str:
    """Repo-relative form when possible, absolute otherwise."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _check_output_dir(module_args: list[str], results: str) -> None:
    """Reject the classic mismatch between ``--results`` and the module's output.

    Only ``--results`` is collected from the VM. A module writing elsewhere
    yields a run that commits nothing but ``colab_run.json`` while the real
    artifacts die with the released VM.
    """
    for index, value in enumerate(module_args):
        declared: str | None = None
        if value == "--output-dir" and index + 1 < len(module_args):
            declared = module_args[index + 1]
        elif value.startswith("--output-dir="):
            declared = value.split("=", 1)[1]
        if declared is None:
            continue
        if Path(declared) != Path(results):
            raise RuntimeError(
                f"Module --output-dir {declared!r} differs from --results {results!r}; "
                "only the --results directory is downloaded from the VM."
            )
        return
    print(
        f"Note: no --output-dir passed to the module. Only {results} is collected "
        "from the VM, so make sure the module writes there."
    )


def _validate_results_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("results",):
        raise ValueError("--results must be a relative path below results/")
    return path


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #


def _account_occupancy() -> dict[str, int]:
    """Sessions this machine believes are still holding a VM, per account."""
    occupancy = {account: 0 for account in colab_accounts.known_accounts()}
    for state in colab_state.active_states():
        account = colab_state.account_of(state)
        occupancy[account] = occupancy.get(account, 0) + 1
    return occupancy


def _select_account(requested: str, max_per_account: int, force: bool) -> str:
    """Pick an account with a free session slot, or validate an explicit choice."""
    occupancy = _account_occupancy()
    if requested != "auto":
        colab_accounts.validate_account(requested)
        if not colab_accounts.is_logged_in(requested):
            raise RuntimeError(
                f"Account {requested!r} is not logged in. Run: "
                f"run_experiment.py login --account {requested}"
            )
        used = occupancy.get(requested, 0)
        if used >= max_per_account and not force:
            raise RuntimeError(
                f"Account {requested!r} already has {used} active session(s); "
                f"cap is {max_per_account}. Use --force to override."
            )
        return requested

    candidates = colab_accounts.logged_in_accounts()
    if not candidates:
        raise RuntimeError(
            "No logged-in Colab accounts. Run: run_experiment.py login --account <name>"
        )
    free = [
        account for account in candidates if occupancy.get(account, 0) < max_per_account
    ]
    if not free:
        summary = ", ".join(
            f"{account}={occupancy.get(account, 0)}" for account in candidates
        )
        raise RuntimeError(
            f"Every account is at the {max_per_account}-session cap ({summary}). "
            "Wait for a session to finish, raise --max-per-account, or add an account."
        )
    return min(free, key=lambda account: (occupancy.get(account, 0), account))


def _colab(
    session: str,
    command: str,
    *args: str,
    account: str,
    timeout: str | None = None,
    local_timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    invocation = ["colab", command, "-s", session]
    if timeout is not None:
        invocation.extend(("--timeout", timeout))
    invocation.extend(args)
    try:
        return _run(
            *invocation,
            capture=True,
            local_timeout=local_timeout,
            env=colab_accounts.account_env(account),
        )
    except subprocess.CalledProcessError as error:
        if error.stdout:
            print(error.stdout, end="")
        if error.stderr:
            print(error.stderr, end="")
        raise


def _colab_plain(account: str, *args: str, check: bool = True) -> str:
    result = _run(
        "colab",
        *args,
        capture=True,
        check=check,
        env=colab_accounts.account_env(account),
    )
    return result.stdout


# --------------------------------------------------------------------------- #
# Source staging
# --------------------------------------------------------------------------- #


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
            raise RuntimeError(
                f"Refusing to upload source containing a credential: {path}"
            )
    with tarfile.open(destination, "w:gz") as archive:
        for path in paths:
            archive.add(PROJECT_ROOT / path, arcname=path)


def _stage_source(state: dict[str, Any]) -> Path:
    """Snapshot the worktree now, so a detached controller uploads what was asked for."""
    directory = colab_state.staging_dir(state["session"])
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    _make_source_archive(directory / "source.tar.gz")
    (directory / "job.json").write_text(json.dumps(state) + "\n", encoding="utf-8")
    return directory


# --------------------------------------------------------------------------- #
# Remote lifecycle
# --------------------------------------------------------------------------- #


def _provision(state: dict[str, Any]) -> None:
    session = state["session"]
    account = colab_state.account_of(state)
    print(f"Provisioning Colab session {session!r} ({state['gpu']}) on {account!r}...")
    _run(
        "colab",
        "new",
        "-s",
        session,
        "--gpu",
        state["gpu"],
        env=colab_accounts.account_env(account),
    )


def _upload_and_start(state: dict[str, Any]) -> None:
    session = state["session"]
    account = colab_state.account_of(state)
    staging = colab_state.staging_dir(session)
    uploads = (
        (staging / "source.tar.gz", "/content/metricdp-source.tar.gz"),
        (staging / "job.json", "/content/metricdp-colab-job.json"),
        (HELPER_DIR / "remote_worker.py", "/content/metricdp-colab-worker.py"),
    )
    for local, remote in uploads:
        _colab(session, "upload", str(local), remote, account=account)
    setup = _colab(
        session, "exec", "-f", str(HELPER_DIR / "remote_setup.py"), account=account,
        timeout="1800",
    )
    print(setup.stdout, end="")
    started = _colab(
        session, "exec", "-f", str(HELPER_DIR / "remote_start.py"), account=account,
        timeout="120",
    )
    print(started.stdout, end="")


def _parse_probe(output: str) -> dict[str, Any]:
    """Split a remote probe into job status, recent log, and a GPU sample."""
    probe: dict[str, Any] = {"raw": output, "state": "unknown", "status": {}}
    marker = "COLAB_JOB_STATUS="
    start = output.find(marker)
    if start >= 0:
        try:
            status, _ = json.JSONDecoder().raw_decode(
                output[start + len(marker) :].lstrip()
            )
        except json.JSONDecodeError:
            status = {}
        if isinstance(status, dict):
            probe["status"] = status
            probe["state"] = str(status.get("state", "unknown"))
    log_start = output.find("--- recent training output ---")
    gpu_start = output.find("--- nvidia-smi ---")
    if log_start >= 0:
        end = gpu_start if gpu_start > log_start else len(output)
        probe["log_tail"] = output[log_start:end].strip()
    if gpu_start >= 0:
        probe["gpu"] = output[gpu_start:].strip()
    return probe


def _probe(session: str, account: str, *, echo: bool = True) -> dict[str, Any]:
    try:
        result = _colab(
            session,
            "exec",
            "-f",
            str(HELPER_DIR / "remote_probe.py"),
            account=account,
            timeout="30",
            local_timeout=45,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        if echo:
            print(
                f"Colab status probe failed ({error}); retrying without stopping training."
            )
        return {"state": "unknown", "status": {}, "raw": "", "error": str(error)}
    if echo:
        print(result.stdout, end="")
    return _parse_probe(result.stdout)


# --------------------------------------------------------------------------- #
# Collection
# --------------------------------------------------------------------------- #


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


def _commit(state: dict[str, Any], remote_state: str) -> str | None:
    """Commit only this session's result directory, serialized against peers.

    Downloading already happened outside the lock; this is the one step every
    concurrent controller contends on, because they share a single Git index.
    """
    results = state["results"]
    with colab_state.commit_lock():
        branch = _git("branch", "--show-current")
        if branch != state["source_branch"]:
            raise RuntimeError(
                f"Current branch changed from {state['source_branch']!r} to {branch!r}; "
                "results were downloaded but were not committed to the wrong branch."
            )
        _run("git", "add", "--", results, capture=True)
        changed = (
            _run(
                "git", "diff", "--cached", "--quiet", "--", results, check=False
            ).returncode
            != 0
        )
        if not changed:
            print(f"No new files under {results}; nothing to commit.")
            return None
        message = state["commit_message"]
        if remote_state != "complete":
            message += " (failed run artifacts)"
        _run("git", "commit", "-m", message, "--", results)
        commit = _git("rev-parse", "--short", "HEAD")
    print(f"Committed {results} as {commit} on {branch} (push manually when ready)")
    return commit


def collect(
    session: str,
    *,
    stop: bool = True,
    remote_state: str | None = None,
) -> str:
    state = colab_state.load_state(session)
    account = colab_state.account_of(state)
    if remote_state is None:
        remote_state = _probe(session, account)["state"]
    if remote_state not in {"complete", "failed"}:
        raise RuntimeError(f"Colab job is not finished (state={remote_state})")
    colab_state.update_state(session, phase=colab_state.PHASE_COLLECTING)
    with tempfile.TemporaryDirectory(prefix="metricdp-colab-download-") as temp_name:
        archive_path = Path(temp_name) / "results.tar.gz"
        _colab(
            session,
            "download",
            "/content/metricdp-colab-results.tar.gz",
            str(archive_path),
            account=account,
        )
        _extract_results(archive_path, Path(state["results"]))
    commit = _commit(state, remote_state)
    colab_state.update_state(
        session,
        phase=colab_state.PHASE_COLLECTED,
        collected_at=colab_state.now(),
        remote_state=remote_state,
        commit=commit,
    )
    _notify_macos(
        "Colab results collected",
        f"Session {session} ({remote_state}) was collected and committed.",
    )
    if stop:
        _colab(session, "stop", account=account)
        print(f"Released Colab session {session!r}")
    shutil.rmtree(colab_state.staging_dir(session), ignore_errors=True)
    return remote_state


# --------------------------------------------------------------------------- #
# Supervision
# --------------------------------------------------------------------------- #


def _supervise(session: str, poll_seconds: int) -> str:
    """Poll a detached remote job, then collect, commit, and release it."""
    account = colab_state.account_of(colab_state.load_state(session))
    while True:
        probe = _probe(session, account)
        if probe["state"] in {"complete", "failed"}:
            break
        time.sleep(poll_seconds)
    return collect(session, remote_state=probe["state"])


def _fail(session: str, error: BaseException) -> None:
    """Record a controller failure, without reopening an already-finished run."""
    current = colab_state.phase(colab_state.load_state(session))
    finished = current in colab_state.TERMINAL_PHASES
    colab_state.update_state(
        session,
        phase=current if finished else colab_state.PHASE_ERROR,
        last_error=f"{type(error).__name__}: {error}",
        failed_at=colab_state.now(),
    )


def launch(session: str, poll_seconds: int) -> None:
    """Provision, upload, start, then supervise one session to completion."""
    state = colab_state.update_state(session, controller_pid=os.getpid())
    try:
        _provision(state)
        colab_state.update_state(session, provisioned_at=colab_state.now())
        _upload_and_start(state)
        colab_state.update_state(
            session,
            phase=colab_state.PHASE_TRAINING,
            training_started_at=colab_state.now(),
        )
        print(f"Training started for {session!r}. Observe with: sweep")
        final_state = _supervise(session, poll_seconds)
        if final_state != "complete":
            raise ExperimentFailed(
                f"Colab experiment {session!r} failed; its artifacts were committed"
            )
    except ExperimentFailed:
        raise
    except KeyboardInterrupt:
        print(
            f"\nLocal monitoring stopped; session {session!r} keeps training. "
            f"`sweep` will adopt and collect it."
        )
        colab_state.update_state(session, controller_pid=None)
        raise
    except Exception as error:
        _fail(session, error)
        print(
            f"Session {session!r} was preserved for recovery. Inspect it with "
            f"`status --session {session}`."
        )
        raise


def _detach(session: str, command: list[str]) -> int:
    """Start a controller in its own session so it survives this shell."""
    log = colab_state.log_path(session)
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("a", encoding="utf-8", buffering=1)
    handle.write(f"\n=== {colab_state.now()} {' '.join(command)} ===\n")
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    return process.pid


def _controller_command(subcommand: str, session: str, poll_seconds: int) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        subcommand,
        "--session",
        session,
        "--poll-seconds",
        str(poll_seconds),
    ]
    # A detached waiter must supervise in its own process, not detach again.
    if subcommand == "wait":
        command.append("--attach")
    return command


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #


def run_job(args: argparse.Namespace) -> None:
    session = colab_state.validate_session(args.session)
    results = _validate_results_path(args.results)
    branch = _git("branch", "--show-current")
    if not branch:
        raise RuntimeError("Colab runs require a named Git branch")
    existing = colab_state.state_path(session)
    if existing.exists():
        previous = colab_state.load_state(session)
        if colab_state.phase(previous) in colab_state.ACTIVE_PHASES:
            raise RuntimeError(
                f"Session {session!r} is already {colab_state.phase(previous)}; "
                "pick another name or collect the existing run first."
            )
    module_args = _forwarded_module_args(args.module_args)
    _check_output_dir(module_args, results.as_posix())
    account = _select_account(args.account, args.max_per_account, args.force)
    state = {
        "session": session,
        "account": account,
        "gpu": args.gpu,
        "module": args.module,
        "args": module_args,
        "results": results.as_posix(),
        "commit_message": args.commit_message,
        "source_branch": branch,
        "source_commit": _git("rev-parse", "HEAD"),
        "phase": colab_state.PHASE_LAUNCHING,
        "launched_at": colab_state.now(),
        "poll_seconds": args.poll_seconds,
        "controller_pid": None,
    }
    colab_state.write_state(state)
    _stage_source(state)

    if args.attach:
        launch(session, args.poll_seconds)
        return
    pid = _detach(session, _controller_command("launch", session, args.poll_seconds))
    print(
        f"session={session} account={account} gpu={args.gpu} pid={pid}\n"
        f"log={_display_path(colab_state.log_path(session))}\n"
        f"Observe every few minutes with: "
        f"uv run python {_display_path(Path(__file__).resolve())} sweep"
    )


def wait_for_job(args: argparse.Namespace) -> None:
    session = colab_state.validate_session(args.session)
    if not args.attach:
        pid = _detach(session, _controller_command("wait", session, args.poll_seconds))
        print(f"Detached waiter for {session} pid={pid}")
        return
    current = colab_state.load_state(session)
    phase = colab_state.phase(current)
    colab_state.update_state(
        session,
        controller_pid=os.getpid(),
        phase=(
            phase
            if phase == colab_state.PHASE_COLLECTING
            else colab_state.PHASE_TRAINING
        ),
    )
    try:
        final_state = _supervise(session, args.poll_seconds)
    except Exception as error:
        _fail(session, error)
        raise
    if final_state != "complete":
        raise ExperimentFailed(
            f"Colab experiment {session!r} failed; its artifacts were committed"
        )


def _age_minutes(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    try:
        moment = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return (datetime.now(UTC) - moment).total_seconds() / 60.0


def _progress(probe: dict[str, Any]) -> tuple[str, str, str]:
    """Round counter, GPU utilization, and last log line from a probe."""
    log_tail = str(probe.get("log_tail", ""))
    rounds = ROUND_PATTERN.findall(log_tail)
    round_text = f"{rounds[-1][0]}/{rounds[-1][1]}" if rounds else "-"
    gpu_match = GPU_PATTERN.search(str(probe.get("gpu", "")))
    gpu_text = f"{gpu_match.group(1)}%" if gpu_match else "-"
    lines = [line.strip() for line in log_tail.splitlines() if line.strip()]
    last_line = lines[-1][:80] if lines else ""
    return round_text, gpu_text, last_line


def _sweep_row(state: dict[str, Any], probe: dict[str, Any]) -> dict[str, Any]:
    session = state["session"]
    phase = colab_state.phase(state)
    alive = colab_state.controller_alive(state)
    round_text, gpu_text, last_line = _progress(probe)
    age = _age_minutes(state.get("launched_at"))
    return {
        "session": session,
        "account": colab_state.account_of(state),
        "phase": phase,
        "remote": probe["state"],
        "round": round_text,
        "gpu": gpu_text,
        "age": f"{age:.0f}m" if age is not None else "-",
        "alive": alive,
        "last_line": last_line,
        "action": "",
        "attention": False,
    }


def _decide(state: dict[str, Any], row: dict[str, Any]) -> tuple[str, str | None]:
    """Classify one session and choose the recovery action ``sweep`` should take."""
    phase = row["phase"]
    remote = row["remote"]
    alive = row["alive"]
    if phase == colab_state.PHASE_ERROR:
        return f"! error: {str(state.get('last_error', ''))[:60]}", None
    if remote in {"complete", "failed"}:
        if alive:
            return f"{remote}; controller collecting", None
        return f"{remote}; collecting", "collect"
    if phase == colab_state.PHASE_LAUNCHING:
        if not alive:
            return "! controller died during launch", None
        if remote == "running":
            return "training", None
        age = _age_minutes(state.get("launched_at")) or 0.0
        if age > LAUNCH_WATCHDOG_MINUTES:
            return f"! no training after {age:.0f}m", None
        return "starting", None
    if remote == "unknown":
        return (
            "! probe failed" if not alive else "probe failed, controller alive"
        ), None
    if phase in {colab_state.PHASE_TRAINING, colab_state.PHASE_COLLECTING}:
        if remote == "not-started":
            # The worker is gone from a VM that was training: Colab recycled it,
            # and anything not yet collected died with it.
            return "! remote worker is gone; VM was reset", None
        if not alive:
            return "! controller gone; waiter relaunched", "adopt"
        stalled = _age_minutes(state.get("last_progress_at"))
        if stalled is not None and stalled > STALL_WATCHDOG_MINUTES:
            return f"! no log progress for {stalled:.0f}m", None
        return "-", None
    return "-", None


def sweep(args: argparse.Namespace) -> None:
    states = [
        state
        for state in colab_state.all_states()
        if colab_state.phase(state) not in colab_state.TERMINAL_PHASES
    ]
    if args.session:
        states = [state for state in states if state["session"] in args.session]
    if not states:
        print("No active Colab sessions.")
        return

    with ThreadPoolExecutor(max_workers=min(8, len(states))) as pool:
        probes = list(
            pool.map(
                lambda state: _probe(
                    state["session"], colab_state.account_of(state), echo=False
                ),
                states,
            )
        )

    rows = []
    for state, probe in zip(states, probes, strict=True):
        row = _sweep_row(state, probe)
        action, task = _decide(state, row)
        row["action"] = action
        row["attention"] = action.startswith("!")
        rows.append(row)

        previous_line = state.get("last_log_line")
        updates: dict[str, Any] = {"last_sweep_at": colab_state.now()}
        if row["last_line"] and row["last_line"] != previous_line:
            updates["last_log_line"] = row["last_line"]
            updates["last_progress_at"] = colab_state.now()
        elif not state.get("last_progress_at"):
            updates["last_progress_at"] = colab_state.now()
        colab_state.update_state(state["session"], **updates)

        if args.dry_run or task is None:
            continue
        if task == "collect":
            try:
                collect(state["session"], remote_state=row["remote"])
                row["action"] = f"{row['remote']}; collected"
            except Exception as error:  # noqa: BLE001 - one bad session must not stop the sweep
                _fail(state["session"], error)
                row["action"] = f"! collect failed: {error}"
                row["attention"] = True
        elif task == "adopt":
            pid = _detach(
                state["session"],
                _controller_command(
                    "wait", state["session"], int(state.get("poll_seconds", 240))
                ),
            )
            colab_state.update_state(state["session"], adopted_at=colab_state.now())
            row["action"] = f"! controller gone; waiter relaunched pid={pid}"

    _print_table(rows)
    attention = [row for row in rows if row["attention"]]
    if attention:
        print(f"\n{len(attention)} session(s) need attention:")
        for row in attention:
            print(f"  {row['session']}: {row['action']}")
            if row["last_line"]:
                print(f"      last log: {row['last_line']}")
            print(f"      log: {_display_path(colab_state.log_path(row['session']))}")


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = ("SESSION", "ACCT", "PHASE", "REMOTE", "ROUND", "GPU", "AGE", "ACTION")
    keys = ("session", "account", "phase", "remote", "round", "gpu", "age", "action")
    widths = [
        max(len(headers[index]), *(len(str(row[key])) for row in rows))
        for index, key in enumerate(keys)
    ]
    print("  ".join(header.ljust(width) for header, width in zip(headers, widths, strict=True)))
    for row in rows:
        print(
            "  ".join(
                str(row[key]).ljust(width)
                for key, width in zip(keys, widths, strict=True)
            )
        )


def show_accounts(args: argparse.Namespace) -> None:
    occupancy = _account_occupancy()
    for account in colab_accounts.known_accounts():
        if colab_accounts.is_logged_in(account):
            email = colab_accounts.resolve_email(account) or "(email unknown)"
        else:
            email = "(not logged in)"
        active = occupancy.get(account, 0)
        print(f"{account:<12} {email:<34} active={active}")
        if args.remote and colab_accounts.is_logged_in(account):
            output = _colab_plain(account, "sessions", check=False).strip()
            for line in output.splitlines():
                print(f"    {line}")


def login(args: argparse.Namespace) -> None:
    account = colab_accounts.validate_account(args.account)
    colab_accounts.prepare_account(account)
    print(
        f"Authorizing account {account!r} with HOME="
        f"{colab_accounts.account_home(account)}.\n"
        "Sign in with the Google account you want this slot to use, then paste "
        "the authorization code below."
    )
    result = subprocess.run(
        ["colab", "sessions"],
        cwd=PROJECT_ROOT,
        env=colab_accounts.account_env(account),
        check=False,
    )
    if result.returncode != 0 or not colab_accounts.is_logged_in(account):
        raise RuntimeError(f"Login for account {account!r} did not produce a token")
    email = colab_accounts.resolve_email(account) or "(email unknown)"
    print(f"Account {account!r} is logged in as {email}")


def stop_session(args: argparse.Namespace) -> None:
    session = colab_state.validate_session(args.session)
    state = colab_state.load_state(session)
    _colab(session, "stop", account=colab_state.account_of(state))
    colab_state.update_state(
        session, phase=colab_state.PHASE_STOPPED, stopped_at=colab_state.now()
    )
    print(f"Released Colab session {session!r}")


def show_status(args: argparse.Namespace) -> None:
    session = colab_state.validate_session(args.session)
    state = colab_state.load_state(session)
    print(
        f"session={session} account={colab_state.account_of(state)} "
        f"phase={colab_state.phase(state)} "
        f"controller={'alive' if colab_state.controller_alive(state) else 'gone'}"
    )
    _probe(session, colab_state.account_of(state))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run", help="stage, provision, train, collect, commit, release"
    )
    run_parser.add_argument("--session", required=True)
    run_parser.add_argument(
        "--gpu", default="L4", choices=("T4", "L4", "G4", "A100", "H100")
    )
    run_parser.add_argument("--module", required=True)
    run_parser.add_argument("--results", required=True)
    run_parser.add_argument("--commit-message", required=True)
    run_parser.add_argument(
        "--account",
        default="auto",
        help="Colab account to use, or 'auto' to pick one with a free slot",
    )
    run_parser.add_argument(
        "--max-per-account", type=int, default=DEFAULT_SESSIONS_PER_ACCOUNT
    )
    run_parser.add_argument(
        "--force", action="store_true", help="ignore the per-account session cap"
    )
    run_parser.add_argument(
        "--attach",
        action="store_true",
        help="block until the run finishes instead of detaching",
    )
    run_parser.add_argument("--poll-seconds", type=int, default=240)
    run_parser.add_argument("module_args", nargs=argparse.REMAINDER)

    launch_parser = subparsers.add_parser(
        "launch", help="drive an already-staged session (used by detached controllers)"
    )
    launch_parser.add_argument("--session", required=True)
    launch_parser.add_argument("--poll-seconds", type=int, default=240)

    wait_parser = subparsers.add_parser(
        "wait", help="poll a running session, then collect and commit it"
    )
    wait_parser.add_argument("--session", required=True)
    wait_parser.add_argument("--poll-seconds", type=int, default=240)
    wait_parser.add_argument(
        "--attach", action="store_true", help="wait in the foreground"
    )

    sweep_parser = subparsers.add_parser(
        "sweep", help="probe every session, auto-collect, and flag what needs a human"
    )
    sweep_parser.add_argument(
        "--session", action="append", help="limit the sweep to these sessions"
    )
    sweep_parser.add_argument(
        "--dry-run", action="store_true", help="report without collecting or adopting"
    )

    accounts_parser = subparsers.add_parser("accounts", help="list configured accounts")
    accounts_parser.add_argument(
        "--remote", action="store_true", help="also list each account's live sessions"
    )

    login_parser = subparsers.add_parser("login", help="authorize one Colab account")
    login_parser.add_argument("--account", required=True)

    for name in ("status", "collect", "stop"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--session", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        run_job(args)
    elif args.command == "launch":
        launch(colab_state.validate_session(args.session), args.poll_seconds)
    elif args.command == "wait":
        wait_for_job(args)
    elif args.command == "sweep":
        sweep(args)
    elif args.command == "accounts":
        show_accounts(args)
    elif args.command == "login":
        login(args)
    elif args.command == "status":
        show_status(args)
    elif args.command == "collect":
        collect(colab_state.validate_session(args.session))
    elif args.command == "stop":
        stop_session(args)


if __name__ == "__main__":
    main()
