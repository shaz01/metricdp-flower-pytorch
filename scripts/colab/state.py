"""Local state files and the commit lock shared by concurrent Colab controllers.

One JSON file per session under ``.colab/`` records everything needed to adopt,
monitor, collect, or diagnose a run that no longer has a live controller.
Downloading and extracting results is per-session and lock-free; only the Git
commit is mutually exclusive, because that is the single resource every
controller shares.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = PROJECT_ROOT / ".colab"
LOG_DIR = STATE_DIR / "logs"
STAGING_DIR = STATE_DIR / "staging"
COMMIT_LOCK_PATH = STATE_DIR / "commit.lock"
SESSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")

# Local lifecycle of one controller, independent of the remote job state.
PHASE_LAUNCHING = "launching"  # provisioning, uploading, installing
PHASE_TRAINING = "training"  # remote worker detached and running
PHASE_COLLECTING = "collecting"  # downloading, extracting, committing
PHASE_COLLECTED = "collected"  # artifacts committed, VM released
PHASE_STOPPED = "stopped"  # VM released without collecting
PHASE_ERROR = "error"  # controller gave up; VM deliberately preserved
ACTIVE_PHASES = (PHASE_LAUNCHING, PHASE_TRAINING, PHASE_COLLECTING)
TERMINAL_PHASES = (PHASE_COLLECTED, PHASE_STOPPED)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def validate_session(session: str) -> str:
    if not SESSION_PATTERN.fullmatch(session):
        raise ValueError("session must contain only letters, digits, '_' or '-'")
    return session


def state_path(session: str) -> Path:
    return STATE_DIR / f"{validate_session(session)}.json"


def log_path(session: str) -> Path:
    return LOG_DIR / f"{validate_session(session)}.log"


def staging_dir(session: str) -> Path:
    return STAGING_DIR / validate_session(session)


def load_state(session: str) -> dict[str, Any]:
    path = state_path(session)
    if not path.exists():
        raise FileNotFoundError(f"No local state for Colab session {session!r}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(state: dict[str, Any]) -> dict[str, Any]:
    """Atomically replace one session's state file."""
    path = state_path(state["session"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(json.dumps(state, indent=2) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)
    return state


def update_state(session: str, **fields: Any) -> dict[str, Any]:
    """Merge fields into an existing state file so distinct writers don't clobber."""
    state = load_state(session)
    state.update(fields)
    return write_state(state)


def all_states() -> list[dict[str, Any]]:
    """Every readable session state, newest launch first."""
    if not STATE_DIR.is_dir():
        return []
    states = []
    for path in sorted(STATE_DIR.glob("*.json")):
        try:
            states.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    states.sort(key=lambda state: str(state.get("launched_at", "")), reverse=True)
    return states


def phase(state: dict[str, Any]) -> str:
    """Phase of a state file, defaulting old pre-phase files to a finished run."""
    return str(state.get("phase", PHASE_COLLECTED))


def active_states() -> list[dict[str, Any]]:
    return [state for state in all_states() if phase(state) in ACTIVE_PHASES]


def account_of(state: dict[str, Any]) -> str:
    return str(state.get("account", "default"))


def controller_alive(state: dict[str, Any]) -> bool:
    """Whether this session's controller process is still running.

    PIDs are recycled, so the command line is checked as well; only a process
    that is still a controller for this session counts as alive.
    """
    pid = state.get("controller_pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return False
    command = result.stdout
    return "run_experiment.py" in command and str(state["session"]) in command


@contextmanager
def commit_lock(timeout: float = 900.0) -> Iterator[None]:
    """Serialize Git commits across every concurrent controller on this machine.

    Collection itself stays outside this lock: Colab VMs disappear without
    warning, so downloads must never queue behind another session's commit.
    """
    COMMIT_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    deadline = datetime.now(UTC).timestamp() + timeout
    with COMMIT_LOCK_PATH.open("a+", encoding="utf-8") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if datetime.now(UTC).timestamp() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting {timeout:.0f}s for {COMMIT_LOCK_PATH}"
                    ) from None
                import time

                time.sleep(1.0)
        try:
            handle.seek(0)
            handle.truncate()
            handle.write(f"{os.getpid()} {now()}\n")
            handle.flush()
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
