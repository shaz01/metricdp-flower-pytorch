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

CONFIG_PATH = Path("/content/metricdp-colab-job.json")
STATUS_PATH = Path("/content/metricdp-colab-status.json")
LOG_PATH = Path("/content/metricdp-colab-training.log")
ARCHIVE_PATH = Path("/content/metricdp-colab-results.tar.gz")
PROJECT_ROOT = Path("/content/metricdp-pytorch")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_status(status: dict[str, object]) -> None:
    temporary = STATUS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATUS_PATH)


def _gpu_snapshot() -> str:
    result = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=False
    )
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
        status.update(
            {
                "state": "complete" if returncode == 0 else "failed",
                "finished_at": _now(),
                "returncode": returncode,
                "error": error,
                "gpu_after": _gpu_snapshot(),
            }
        )
        shutil.copy2(LOG_PATH, result_dir / "colab_training.log")
        (result_dir / "colab_run.json").write_text(
            json.dumps(status, indent=2) + "\n", encoding="utf-8"
        )
        _write_status(status)
        with tarfile.open(ARCHIVE_PATH, "w:gz") as archive:
            archive.add(result_dir, arcname=config["results"])


if __name__ == "__main__":
    main()
