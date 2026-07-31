"""Sweep 48-client FedYogi at noise multipliers calibrated to a target noise sigma.

Averaging more clients lowers each client's sensitivity, so the standard
DP-FedAvg Gaussian calibration injects noise of
``sigma = noise_multiplier * clipping_norm / num_sampled_clients`` (McMahan et
al. 2018, arXiv:1710.06963; implemented in
``flwr.supercore.differential_privacy.compute_stdv``). The multiplier is the
privacy parameter, so a fixed value carries the same guarantee at any cohort
size -- but the perturbation it actually applies shrinks as the cohort grows,
and with it the measurable utility gap between the privacy arms. Comparing runs
across client counts therefore has to hold sigma fixed rather than the
multiplier. This sweep targets sigma directly by inverting that relation,
``nm = sigma * N / C``:

=========  ==========================  ====================
sigma      regime seen at 8 clients    nm at N=48, C=5.0
=========  ==========================  ====================
1.25e-02   edge of threshold           0.12
3.12e-02   the gap window              0.30
6.25e-02   collapse                    0.60
=========  ==========================  ====================

Note that these multipliers are far above the paper's 0.01, so these runs sit
at a strictly stronger guarantee than the paper's operating point. They are
chosen for measurability at 48 clients, not to reproduce the paper's setting.

Design notes
------------
Carried on ``fedyogi`` rather than ``fedavg``: at 48 clients plain FedAvg does
not converge within the paper's 20 rounds even without privacy (vanilla reaches
only 0.6094), so a privacy gap measured on top of it would be confounded by an
unconverged baseline. FedYogi reaches 0.9109 vanilla at the same budget.

``vanilla`` ignores ``noise_multiplier`` and ``clipping_norm`` entirely, so it
is run once per partition as the reference rather than once per sigma.

Runs are forced offline against the local HuggingFace cache. ``client.py``
rebuilds the data module per client per round, so a 48-client 20-round run makes
~960 ``load_dataset`` calls; each one revalidates against the Hub, and 12
concurrent unauthenticated actors get rate-limited into multi-minute backoff.
Measured cold load: 155.94s online vs 0.10s offline. ``--allow-hf-network``
opts out, and a preflight check fails fast if the cache cannot serve the
dataset offline.

The runner automatically shares one CUDA device across the concurrently
scheduled client actors (and uses 0.0 GPU resources when CUDA is unavailable).
At 48 clients / 12 parallel actors, this assigns each actor a logical share of
0.0825. GPU-trained results are not bit-comparable with the earlier CPU-only
sweeps, which is why this sweep carries its own vanilla reference rather than
reusing ``results/48client_scaling``.

Reuses ``experiments.reproduce.runner`` unmodified via subprocess, exactly like
``sweep_48_clients.py``: resumable (skips combinations whose result JSON already
reports the expected number of completed rounds), continues past a failing
combination rather than aborting the whole multi-hour sweep, and supports
``--force`` to ignore existing results and rerun everything.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from experiments.reproduce.matrix import Combo, Hyperparams
from experiments.reproduce.matrix.run_combo import run_one_combo

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NUM_CLIENTS = 48
CLIPPING_NORM = 5.0  # runner default; kept explicit since sigma depends on it
AGGREGATION = "fedyogi"
PARTITION_MODES = ("homogeneous", "non-iid")
DP_PRIVACY_MODES = ("global-dp", "metric-privacy")
SEED = 42
ROUNDS = 20
LOCAL_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
INITIALIZATION_EPOCHS = 20

# nm = sigma * N / C, chosen to reproduce the three regimes found at 8 clients.
NOISE_MULTIPLIERS = (0.12, 0.30, 0.60)

MAX_PARALLEL_CLIENTS = 12
DEFAULT_NOISE_MULTIPLIER = 0.01

# Pin every subprocess to the local HF cache; see module docstring.
HF_OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
OUTPUT_DIR = PROJECT_ROOT / "results" / "sigma_calibration"
LOG_PATH = OUTPUT_DIR / "sweep_progress.log"


def effective_sigma(noise_multiplier: float) -> float:
    """Return the per-element noise stdev actually injected for a multiplier.

    The standard DP-FedAvg calibration, ``noise_multiplier * clipping_norm /
    num_sampled_clients``, as implemented by
    ``flwr.supercore.differential_privacy.compute_stdv``. For ``metric-privacy``
    this is the *base* sigma before the strategy's ``noise_multiplier /
    distance`` recalibration, so the realised value drifts with the logged
    ``metric-dp-distance``.
    """
    return noise_multiplier * CLIPPING_NORM / NUM_CLIENTS


def subprocess_env(*, offline: bool) -> dict[str, str]:
    """Return the child environment, optionally pinned to the local HF cache."""
    env = os.environ.copy()
    if offline:
        env.update(HF_OFFLINE_ENV)
    return env


def preflight_dataset_cache(*, offline: bool) -> None:
    """Fail fast if the dataset cannot be materialised under the chosen mode.

    Running offline against a cold cache would otherwise fail 14 times in a row,
    once per combination, after paying Ray startup each time.
    """
    probe = [
        sys.executable,
        "-c",
        "from experiments.reproduce.dataset.alzheimer import load_alzheimer_dataset;"
        " d = load_alzheimer_dataset();"
        " print({k: len(v) for k, v in d.items()})",
    ]
    result = subprocess.run(
        probe,
        cwd=PROJECT_ROOT,
        env=subprocess_env(offline=offline),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        mode = "offline" if offline else "online"
        raise SystemExit(
            f"Preflight dataset load failed ({mode}). Warm the cache once with\n"
            f"  uv run python -m experiments.client_scaling.sweep_48_sigma_calibrated "
            f"--allow-hf-network\n"
            f"or pre-download the dataset, then rerun.\n\n{result.stderr.strip()[-2000:]}"
        )
    _log(f"Preflight dataset load OK (offline={offline}): {result.stdout.strip()}")


def _log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def iter_combos() -> list[tuple[str, str, float | None]]:
    """Enumerate ``(partition, privacy, noise_multiplier)`` in execution order.

    The vanilla reference for a partition runs first so that a partially
    completed sweep still yields an interpretable baseline.
    """
    combos: list[tuple[str, str, float | None]] = []
    for partition in PARTITION_MODES:
        combos.append((partition, "vanilla", None))
        for noise_multiplier in NOISE_MULTIPLIERS:
            for privacy in DP_PRIVACY_MODES:
                combos.append((partition, privacy, noise_multiplier))
    return combos


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun every combination even if a complete result already exists",
    )
    parser.add_argument(
        "--max-parallel-clients",
        type=int,
        default=MAX_PARALLEL_CLIENTS,
        help="cap simultaneous Ray actors to control memory use",
    )
    parser.add_argument(
        "--allow-hf-network",
        action="store_true",
        help="permit HuggingFace Hub access instead of pinning to the local cache",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    offline = not args.allow_hf_network
    preflight_dataset_cache(offline=offline)
    combos = iter_combos()
    sigma_grid = ", ".join(
        f"nm={nm:g}->sigma={effective_sigma(nm):.3e}" for nm in NOISE_MULTIPLIERS
    )
    _log(
        f"Sweep starting: {len(combos)} combinations, num_clients={NUM_CLIENTS}, "
        f"aggregation={AGGREGATION}, clipping_norm={CLIPPING_NORM}, seed={SEED}, "
        f"grid=[{sigma_grid}], max_parallel_clients={args.max_parallel_clients}, "
        f"hf_offline={offline}, force={args.force}"
    )

    completed = 0
    failed: list[str] = []
    for partition, privacy, noise_multiplier in combos:
        combo = Combo(
            name_prefix="sigma48",
            num_clients=NUM_CLIENTS,
            partition=partition,
            privacy=privacy,
            aggregation=AGGREGATION,
            seed=SEED,
            hyperparams=Hyperparams(
                noise_multiplier=(
                    noise_multiplier
                    if noise_multiplier is not None
                    else DEFAULT_NOISE_MULTIPLIER
                ),
                clipping_norm=CLIPPING_NORM,
                rounds=ROUNDS,
                local_epochs=LOCAL_EPOCHS,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                initialization_epochs=INITIALIZATION_EPOCHS,
            ),
        )
        if noise_multiplier is None:
            start_detail = "(no noise)"
        else:
            start_detail = (
                f"(nm={noise_multiplier}, "
                f"sigma={effective_sigma(noise_multiplier):.3e})"
            )
        ok = run_one_combo(
            combo,
            output_dir=OUTPUT_DIR,
            max_parallel_clients=args.max_parallel_clients,
            force=args.force,
            log=_log,
            env=subprocess_env(offline=offline),
            start_detail=start_detail,
        )
        completed += 1
        if not ok:
            failed.append(combo.run_name())
        _log(f"PROGRESS {completed}/{len(combos)} ({len(failed)} failed so far)")

    _log(f"Sweep finished: {completed}/{len(combos)} attempted, {len(failed)} failed")
    if failed:
        _log("Failed combinations: " + ", ".join(failed))


if __name__ == "__main__":
    main()
