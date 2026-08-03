"""Small logging helpers shared by experiment runners."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

LogFunction = Callable[[str], None]


def make_file_logger(log_path: Path) -> LogFunction:
    """Return a logger that timestamps messages to stdout and a text file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return log
