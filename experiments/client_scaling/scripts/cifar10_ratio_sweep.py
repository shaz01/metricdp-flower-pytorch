"""Stage A: accuracy-only noise-ratio calibration for 10-class CIFAR-10.

``PLAN.md``'s noise-ratio system fixes ``noise_multiplier = ratio *
num_clients``, so that Flower's global-DP noise standard deviation
(``nm * clipping_norm / n``) stays at ``ratio * clipping_norm`` no matter how
many clients participate.  The ratios committed under
``results/planned_runs/cifar`` (0.0025, 0.003333, 0.00625) were calibrated on
the **4-class** CIFAR model; this repo's scaling work has since moved to the
full 10-class model, whose loss scale and update norms differ, so those ratios
cannot simply be carried over.  This sweep re-locates the usable band on the
10-class model at the client counts the CIA experiment will use.

It is deliberately accuracy-only: no shadow split, no checkpoints, no attack.
The point is to find, per client count, the ratio at which the two mechanisms
separate without either collapsing, before spending the much larger CIA budget
(Stage B).

**Two arms, because a fixed ratio does not mean equal noise.**  Global-DP's
noise depends only on the ratio, but metric-privacy first divides the
multiplier by the measured maximum pairwise client distance ``d``
(``metricdp_strategy.py``), so it injects ``ratio * clipping_norm / d``.  Since
``d`` shrinks as clients are added -- measured at roughly 1.6-2.0 at n=3 down
to 0.81-0.87 at n=48 on the 4-class runs, see
``results/client_scaling/noise_scaling_diagnostics.json`` -- a fixed ratio
quietly gives metric-privacy less noise than global-DP at small ``n`` and more
at large ``n``:

* ``--arm fixed-ratio`` (default) reproduces PLAN.md's system as written, and
  stays comparable with every ratio run already committed.
* ``--arm matched-noise`` multiplies metric-privacy's multiplier by a measured
  ``--distance``, so both mechanisms inject the same noise standard deviation
  and any accuracy difference is attributable to where the noise goes rather
  than how much of it there is.  Obtain ``--distance`` from the fixed-ratio
  arm's own ``metric-dp-distance`` diagnostic at the same client count.

Runs use the IN-replace participant view, matching the accuracy-only
calibration already committed under ``results/planned_runs/cifar`` and the
adjacency Stage B's CIA will use, so a calibrated ratio transfers directly.

One chunk per session keeps Colab units small:

    uv run python -m experiments.client_scaling.scripts.cifar10_ratio_sweep \\
      --clients 48 --ratio 0.00625 --privacy metric-privacy
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from experiments.cia.datasets.partitions import PartitionViewDataModule, in_replace
from experiments.client_scaling.sweep_runner import run_sweep
from experiments.reproduce.dataset.cifar10 import Cifar10DataModule
from experiments.reproduce.matrix import Combo, Hyperparams

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "client_scaling" / "cifar10_ratio_sweep"
)

CLIENT_COUNTS = (8, 48, 100)
# 0.0025 is the floor already characterised on the 4-class runs (both mechanisms
# healthy); 0.00625 is where they separated at n=3 and where the 10-class
# homogeneous sweep saw a 19.5pp gap at n=8; 0.004 samples the steep region
# between them. 0.0125 is deliberately excluded -- it already collapsed
# global-DP to 0.190 accuracy at n=4 on the 10-class model, and a collapsed
# baseline makes the metric-vs-global comparison meaningless.
RATIOS = (0.0025, 0.004, 0.00625)
PRIVACY_MODES = ("vanilla", "global-dp", "metric-privacy")
ARMS = ("fixed-ratio", "matched-noise")
PARTITION_MODE = "non-iid"
AGGREGATION = "fedavg"
SEED = 42  # single calibration seed, as in PLAN.md's noise-sweep section
TARGET_PARTITION_ID = 0

ROUNDS = 20
HYPERPARAMS = Hyperparams(
    clipping_norm=5.0,
    rounds=ROUNDS,
    local_epochs=5,
    batch_size=32,
    learning_rate=0.001,
    initialization_epochs=20,
)
# Vanilla ignores the multiplier; this keeps its run name stable across ratios.
VANILLA_NOISE_MULTIPLIER = 0.01

MODEL_MODULE = "experiments.reproduce.cifar10_cnn:create_model"
DATA_MODULE = "experiments.client_scaling.scripts.cifar10_ratio_sweep:create_in_replace"


def _cache_dir(config: Mapping[str, Any]) -> str | None:
    return str(config.get("data-cache-dir", "")).strip() or None


def create_in_replace(config: Mapping[str, Any]) -> PartitionViewDataModule:
    """IN-replace view: canonical partitions minus the replacement client.

    Replacement adjacency keeps the active client count equal on both sides of
    the eventual IN/OUT comparison, which is exactly why PLAN.md defines the
    noise ratio only for replace settings -- a single ratio maps to one
    multiplier for both views.
    """
    active_clients = int(config["num-clients"])
    if active_clients < 2:
        raise ValueError("Replacement adjacency requires at least two active clients.")
    canonical_clients = active_clients + 1
    return in_replace(
        Cifar10DataModule(cache_dir=_cache_dir(config)),
        canonical_num_partitions=canonical_clients,
        target_partition_id=TARGET_PARTITION_ID,
        replacement_partition_id=canonical_clients - 1,
    )


def noise_multiplier_for(
    *,
    privacy: str,
    ratio: float,
    num_clients: int,
    arm: str = "fixed-ratio",
    distance: float | None = None,
) -> float:
    """Map a noise ratio to this run's multiplier.

    ``fixed-ratio`` is PLAN.md's definition, ``nm = ratio * n``.
    ``matched-noise`` additionally rescales metric-privacy by ``distance`` so
    that its injected standard deviation matches global-DP's at the same ratio,
    cancelling the ``nm / d`` calibration the mechanism applies internally.
    """
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}.")
    if privacy == "vanilla":
        return VANILLA_NOISE_MULTIPLIER
    if ratio <= 0:
        raise ValueError("ratio must be positive for a non-vanilla run.")
    multiplier = ratio * num_clients
    if arm == "matched-noise" and privacy == "metric-privacy":
        if distance is None or distance <= 0:
            raise ValueError(
                "matched-noise requires a positive --distance for metric-privacy."
            )
        multiplier *= distance
    return multiplier


def build_combos(
    *,
    num_clients: int,
    ratios: Sequence[float],
    privacy_modes: Sequence[str] = PRIVACY_MODES,
    arm: str = "fixed-ratio",
    distance: float | None = None,
    seed: int = SEED,
) -> list[Combo]:
    """Build one accuracy-only combo per (ratio, privacy) pair."""
    if num_clients < 2:
        raise ValueError("num_clients must be at least 2.")
    for privacy in privacy_modes:
        if privacy not in PRIVACY_MODES:
            raise ValueError(f"privacy mode must come from {PRIVACY_MODES}.")

    combos = []
    for privacy in privacy_modes:
        # Vanilla is ratio-independent: run it once, not once per ratio.
        run_ratios = ratios[:1] if privacy == "vanilla" else ratios
        for ratio in run_ratios:
            combos.append(
                Combo(
                    name_prefix=f"cifar10-ratio-{arm}",
                    num_clients=num_clients,
                    partition=PARTITION_MODE,
                    privacy=privacy,
                    aggregation=AGGREGATION,
                    seed=seed,
                    noise_multiplier=noise_multiplier_for(
                        privacy=privacy,
                        ratio=ratio,
                        num_clients=num_clients,
                        arm=arm,
                        distance=distance,
                    ),
                    hyperparams=HYPERPARAMS,
                    data_module=DATA_MODULE,
                    model_module=MODEL_MODULE,
                )
            )
    return combos


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clients",
        type=int,
        required=True,
        help=f"Active client count for this chunk (planned: {CLIENT_COUNTS}).",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        nargs="+",
        default=list(RATIOS),
        help=f"Noise ratios to sweep (default: {RATIOS}).",
    )
    parser.add_argument(
        "--privacy",
        choices=PRIVACY_MODES,
        nargs="+",
        default=list(PRIVACY_MODES),
        help="Privacy modes; pass one to split a session further.",
    )
    parser.add_argument(
        "--arm",
        choices=ARMS,
        default="fixed-ratio",
        help="fixed-ratio reproduces PLAN.md; matched-noise equalises the "
        "injected noise across mechanisms (needs --distance).",
    )
    parser.add_argument(
        "--distance",
        type=float,
        help="Measured metric-dp-distance at this client count, for "
        "--arm matched-noise.",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--max-parallel-clients",
        type=int,
        help="Defaults to min(clients, 8).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    combos = build_combos(
        num_clients=args.clients,
        ratios=args.ratio,
        privacy_modes=args.privacy,
        arm=args.arm,
        distance=args.distance,
        seed=args.seed,
    )
    output_dir = (args.output_dir / f"clients-{args.clients}").resolve()
    run_sweep(
        combos,
        output_dir=output_dir,
        log_path=output_dir / "progress.log",
        max_parallel_clients=args.max_parallel_clients or min(args.clients, 8),
        force=args.force,
        start_message=(
            f"CIFAR-10 ratio sweep ({args.arm}): {len(combos)} runs, "
            f"clients={args.clients}, ratios={args.ratio}"
        ),
    )


if __name__ == "__main__":
    main()
