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

Dataset loading uses the process-wide singleton in
``experiments.reproduce.dataset.alzheimer``. Its first access prefers the local
HuggingFace cache and falls back to the network on a cold cache; later accesses
reuse the materialised dataset without contacting the Hub.

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
import time
from pathlib import Path

from experiments.reproduce.matrix import Combo, Hyperparams, Matrix
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


def _log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _matrix(
    partition: str,
    privacy_modes: tuple[str, ...],
    noise_multipliers: tuple[float, ...],
) -> Matrix:
    """Return one partition's matrix over the requested noise multipliers."""
    return Matrix(
        partitions=(partition,),
        privacy_modes=privacy_modes,
        aggregations=(AGGREGATION,),
        seeds=(SEED,),
        noise_multipliers=noise_multipliers,
        hyperparams=Hyperparams(
            clipping_norm=CLIPPING_NORM,
            rounds=ROUNDS,
            local_epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            initialization_epochs=INITIALIZATION_EPOCHS,
        ),
    )


def iter_combos() -> list[Combo]:
    """Enumerate every configured run in execution order.

    The noise multiplier is a matrix dimension, with one matrix restricted to
    a single partition at a time. The vanilla reference for a partition runs
    first -- and only once, since it ignores the noise settings entirely -- so
    that a partially completed sweep still yields an interpretable baseline.
    """
    combos: list[Combo] = []
    for partition in PARTITION_MODES:
        matrices = [_matrix(partition, ("vanilla",), (DEFAULT_NOISE_MULTIPLIER,))]
        matrices += [_matrix(partition, DP_PRIVACY_MODES, NOISE_MULTIPLIERS)]
        for matrix in matrices:
            combos.extend(
                matrix.list_combos(name_prefix="sigma48", num_clients=NUM_CLIENTS)
            )
    return combos


def start_detail(combo: Combo) -> str:
    """Describe a combo's noise setting for the launch log line."""
    if combo.privacy == "vanilla":
        return "(no noise)"
    noise_multiplier = combo.noise_multiplier
    return f"(nm={noise_multiplier}, sigma={effective_sigma(noise_multiplier):.3e})"


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
    return parser


def main() -> None:
    args = _parser().parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    combos = iter_combos()
    sigma_grid = ", ".join(
        f"nm={nm:g}->sigma={effective_sigma(nm):.3e}" for nm in NOISE_MULTIPLIERS
    )
    _log(
        f"Sweep starting: {len(combos)} combinations, num_clients={NUM_CLIENTS}, "
        f"aggregation={AGGREGATION}, clipping_norm={CLIPPING_NORM}, seed={SEED}, "
        f"grid=[{sigma_grid}], max_parallel_clients={args.max_parallel_clients}, "
        f"force={args.force}"
    )

    completed = 0
    failed: list[str] = []
    for combo in combos:
        ok = run_one_combo(
            combo,
            output_dir=OUTPUT_DIR,
            max_parallel_clients=args.max_parallel_clients,
            force=args.force,
            log=_log,
            start_detail=start_detail(combo),
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
