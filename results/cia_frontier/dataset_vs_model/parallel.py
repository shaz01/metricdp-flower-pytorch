"""Run several stage_b invocations side by side on one machine (one Colab VM).

A single trajectory leaves an L4 VM mostly idle (GPU ~5-30%, 6 of 12 CPUs), so a
shard can run two independent streams of trajectories at once:

    python -m results.cia_frontier.dataset_vs_model.parallel <common stage_b args> \
        ::: <stream 1 args> ::: <stream 2 args>

Each stream is ``python -m results.cia_frontier.dataset_vs_model.stage_b <common> <stream>``.
Streams must train disjoint trajectories (stage_b's per-trajectory lock enforces it).
Exits nonzero if any stream fails, after all streams have finished.
"""
from __future__ import annotations

import subprocess
import sys

SEPARATOR = ":::"
MODULE = "results.cia_frontier.dataset_vs_model.stage_b"


def split_streams(argv: list[str]) -> tuple[list[str], list[list[str]]]:
    if SEPARATOR not in argv:
        raise ValueError(f"Give at least one '{SEPARATOR}' stream")
    first = argv.index(SEPARATOR)
    common, streams, current = argv[:first], [], None
    for token in argv[first:]:
        if token == SEPARATOR:
            current = []
            streams.append(current)
        else:
            current.append(token)
    if any(not s for s in streams):
        raise ValueError("Empty stream")
    return common, streams


def commands(argv: list[str], python: str = sys.executable) -> list[list[str]]:
    common, streams = split_streams(argv)
    return [[python, "-u", "-m", MODULE, *common, *stream] for stream in streams]


def main(argv: list[str] | None = None) -> int:
    processes = [subprocess.Popen(c) for c in commands(sys.argv[1:] if argv is None else argv)]
    codes = [p.wait() for p in processes]
    print(f"[PARALLEL] stream exit codes: {codes}", flush=True)
    return 0 if all(c == 0 for c in codes) else 1


if __name__ == "__main__":
    sys.exit(main())
