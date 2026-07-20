"""Compare the original Flower 1.16 strategy with the Flower 1.32 port."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ORIGINAL = PROJECT / "experiments/original_metricdp_fixed_clipping.py"
SEEDS = (0, 1, 42, 123456, 20260721)


def run(command: list[str]) -> dict:
    """Run a comparison worker and parse its final JSON output line."""
    completed = subprocess.run(
        command,
        cwd=PROJECT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def main() -> None:
    """Run both implementations with identical inputs and RNG seeds."""
    if not ORIGINAL.exists():
        raise FileNotFoundError(f"Original strategy not found at {ORIGINAL}")

    all_within_tolerance = True
    for seed in SEEDS:
        legacy = run(
            [
                "uv",
                "run",
                "--quiet",
                "--isolated",
                "--python",
                "3.12",
                "--with",
                "flwr==1.16.0",
                "--with",
                "numpy==1.26.4",
                "--with",
                "scipy==1.14.1",
                "experiments/legacy_runner.py",
                str(ORIGINAL),
                str(seed),
            ]
        )
        modern = run([sys.executable, "experiments/modern_runner.py", str(seed)])

        old_arrays = [np.asarray(array) for array in legacy["arrays"]]
        new_arrays = [np.asarray(array) for array in modern["arrays"]]
        output_diff = max(
            float(np.max(np.abs(old - new)))
            for old, new in zip(old_arrays, new_arrays, strict=True)
        )
        distance_diff = abs(legacy["distance"] - modern["distance"])
        stdv_diff = abs(legacy["stdv"] - modern["stdv"])
        within_tolerance = max(output_diff, distance_diff, stdv_diff) <= 1e-15
        all_within_tolerance &= within_tolerance
        print(
            f"seed={seed:<8} distance_diff={distance_diff:.1e} "
            f"stdv_diff={stdv_diff:.1e} output_max_diff={output_diff:.1e} "
            f"within_1e-15={within_tolerance}"
        )

    if not all_within_tolerance:
        raise SystemExit("Implementations differ beyond floating-point tolerance")


if __name__ == "__main__":
    main()
