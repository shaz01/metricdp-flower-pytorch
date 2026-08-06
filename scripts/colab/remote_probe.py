"""Print job state, recent training output, and current GPU use."""

from __future__ import annotations

import subprocess
from pathlib import Path

STATUS_PATH = Path("/content/metricdp-colab-status.json")
LOG_PATH = Path("/content/metricdp-colab-training.log")


def main() -> None:
    if STATUS_PATH.exists():
        print("COLAB_JOB_STATUS=" + STATUS_PATH.read_text(encoding="utf-8").strip())
    else:
        print('COLAB_JOB_STATUS={"state": "not-started"}')
    print("\n--- recent training output ---")
    if LOG_PATH.exists():
        print("".join(LOG_PATH.read_text(errors="replace").splitlines(True)[-20:]))
    print("--- nvidia-smi ---")
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
    print(gpu.stdout if gpu.returncode == 0 else gpu.stderr)


if __name__ == "__main__":
    main()
