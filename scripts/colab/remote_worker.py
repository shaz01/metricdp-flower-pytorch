"""Background supervisor for one Colab experiment module."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import traceback
from datetime import UTC, datetime
from pathlib import Path

# Colab always mounts /content; the override exists so the remote half can be
# exercised locally by the end-to-end test without a real VM.
CONTENT = Path(os.environ.get("METRICDP_COLAB_CONTENT", "/content"))
CONFIG_PATH = CONTENT / "metricdp-colab-job.json"
STATUS_PATH = CONTENT / "metricdp-colab-status.json"
LOG_PATH = CONTENT / "metricdp-colab-training.log"
ARCHIVE_PATH = CONTENT / "metricdp-colab-results.tar.gz"
PROJECT_ROOT = CONTENT / "metricdp-pytorch"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_status(status: dict[str, object]) -> None:
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _gpu_snapshot() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        )
    except OSError as error:  # CPU runtime, or a VM without the driver
        return f"nvidia-smi unavailable: {error}"
    return result.stdout if result.returncode == 0 else result.stderr


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    result_dir = PROJECT_ROOT / config["results"]
    result_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-u", "-m", config["module"], *config["args"]]
    status: dict[str, object] = {
        "state": "starting",
        "started_at": _now(),
        "command": command,
        "source_commit": config["source_commit"],
        "source_branch": config["source_branch"],
        "results": config["results"],
        "python": sys.version,
        "platform": platform.platform(),
        "gpu_before": _gpu_snapshot(),
    }
    _write_status(status)
    returncode = 1
    error: str | None = None
    try:
        with LOG_PATH.open("w", encoding="utf-8", buffering=1) as log:
            process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env={
                    **os.environ,
                    "METRICDP_SOURCE_COMMIT": config["source_commit"],
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "PYTHONHASHSEED": "0",
                },
            )
            status.update({"state": "running", "pid": process.pid})
            _write_status(status)
            returncode = process.wait()
    except Exception:  # noqa: BLE001 - preserve diagnostics and package outputs
        error = traceback.format_exc()
    finally:
        final_state = "complete" if returncode == 0 else "failed"
        status.update(
            {
                "state": "packaging",
                "finished_at": _now(),
                "returncode": returncode,
                "error": error,
                "gpu_after": _gpu_snapshot(),
            }
        )
        _write_status(status)
        shutil.copy2(LOG_PATH, result_dir / "colab_training.log")
        (result_dir / "colab_run.json").write_text(
            json.dumps(status | {"state": final_state}, indent=2) + "\n",
            encoding="utf-8",
        )
        # Publish the archive atomically and only then announce the terminal
        # state: a controller that collects on "complete" must never download a
        # tarball that is still being written.
        staging = ARCHIVE_PATH.with_suffix(".partial")
        with tarfile.open(staging, "w:gz") as archive:
            archive.add(result_dir, arcname=config["results"])
        staging.replace(ARCHIVE_PATH)
        status.update(
            {
                "state": final_state,
                "archived_at": _now(),
                "archive_bytes": ARCHIVE_PATH.stat().st_size,
            }
        )
        _write_status(status)


if __name__ == "__main__":
    main()
