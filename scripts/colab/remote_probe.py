"""Print job state, recent training output, and current GPU use."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

CONTENT = Path(os.environ.get("METRICDP_COLAB_CONTENT", "/content"))
STATUS_PATH = CONTENT / "metricdp-colab-status.json"
LOG_PATH = CONTENT / "metricdp-colab-training.log"
ARCHIVE_PATH = CONTENT / "metricdp-colab-results.tar.gz"


def main() -> None:
    if STATUS_PATH.exists():
        print("COLAB_JOB_STATUS=" + STATUS_PATH.read_text(encoding="utf-8").strip())
    else:
        print('COLAB_JOB_STATUS={"state": "not-started"}')
    size = ARCHIVE_PATH.stat().st_size if ARCHIVE_PATH.exists() else 0
    print(f"\n--- results archive --- {size} bytes")
    print("\n--- recent training output ---")
    if LOG_PATH.exists():
        print("".join(LOG_PATH.read_text(errors="replace").splitlines(True)[-20:]))
    print("--- nvidia-smi ---")
    try:
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:  # CPU runtime, or a VM without the driver
        print(f"nvidia-smi unavailable: {error}")
        return
    print(gpu.stdout if gpu.returncode == 0 else gpu.stderr)


if __name__ == "__main__":
    main()
