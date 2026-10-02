"""Detach the Colab experiment worker from the notebook kernel."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CONTENT = Path(os.environ.get("METRICDP_COLAB_CONTENT", "/content"))
WORKER = CONTENT / "metricdp-colab-worker.py"


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
