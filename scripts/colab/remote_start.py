"""Detach the Colab experiment worker from the notebook kernel."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

WORKER = Path("/content/metricdp-colab-worker.py")


def main() -> None:
    process = subprocess.Popen(
        [sys.executable, str(WORKER)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"Started Colab experiment supervisor pid={process.pid}")


if __name__ == "__main__":
    main()
