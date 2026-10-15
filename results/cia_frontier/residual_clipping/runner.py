"""WP6 (D4 item 2) residual-clipping screen. Plans unless --execute.

Design (PROTOCOL.md): EuroSAT, label-Dirichlet alpha 0.3, 48 slots, seed 42, 50 rounds, removal adjacency (one IN
trajectory, one OUT trajectory per target), the five top-decile clients as targets, equal weights and the public slot
count as denominator, four mechanisms at the same per-round guarantee (1/z_plain-GDP), WP0 diagnostics on.

Several workers may run the same command: each trajectory is locked while it trains (WP2's stall-guarded
``execute_shared``), locked ones are skipped, completed ones are skipped on restart.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import json
import math
from pathlib import Path

from results.cia_frontier.eurosat_frontier.runner import FrontierCombo, atomic_json, build_combos
from results.contest_at_scale.auc_frontier.eurosat_remove import HYPERPARAMS

ALPHA = 0.3
SLOTS = 48
SEEDS = (42,)
LAYOUT_SEED = 42
ROUNDS = 50
# Fixed before any run: the top decile (ceil(0.1 x 48) = 5 largest clients) of the seed-42 partition.
TARGETS = (23, 7, 40, 18, 33)
RATIO = 0.001546  # the committed frontier noise: plain clipping sigma = RATIO * C on the average
CLIP = HYPERPARAMS.clipping_norm  # 5.0
Z_PLAIN = RATIO * SLOTS  # sigma = z C / n  ->  z = RATIO * n; per-round mu = 1 / z
QUANTILE_SHARE = 0.1  # share of each round's mu^2 spent on the private rho (as in WP3)
Z_RESIDUAL = Z_PLAIN / math.sqrt(1 - QUANTILE_SHARE)
BIT_NOISE_STD = Z_PLAIN / math.sqrt(QUANTILE_SHARE)
RHO_BOUNDS = (0.05, 4 * CLIP)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "cia_frontier" / "residual_clipping" / "results" / "stage_a"
CONFIG_SCHEMA = 1

# name -> (center, fixed rho or None, target quantile or None, noise multiplier)
CONFIGS = {
    "plain": ("zero", CLIP, None, Z_PLAIN),
    "residual-fixed": ("previous", CLIP, None, Z_PLAIN),
    "residual-median": ("previous", None, 0.5, Z_RESIDUAL),
    "residual-q90": ("previous", None, 0.9, Z_RESIDUAL),
}
CANDIDATES = ("residual-median", "residual-q90")
TRAJECTORY_HOURS_PER_ROUND = 0.74 / 100


def mu_per_round(config):
    _, rho, quantile, z = CONFIGS[config]
    return math.hypot(1 / z, 1 / BIT_NOISE_STD) if quantile is not None else 1 / z


@dataclass(frozen=True, kw_only=True)
class ResidualCombo(FrontierCombo):
    config: str
    center: str
    rho: float | None
    quantile: float | None

    def runner_args(self, **kwargs):
        args = (*super().runner_args(**kwargs), "--num-slots", str(SLOTS), "--residual-center", self.center)
        if self.quantile is None:
            return (*args, "--residual-rho", repr(self.rho))
        return (*args, "--residual-quantile", repr(self.quantile), "--residual-rho-min", repr(RHO_BOUNDS[0]),
                "--residual-rho-max", repr(RHO_BOUNDS[1]), "--residual-bit-noise-std", repr(BIT_NOISE_STD))


def build_residual_combos(config, seeds=SEEDS, targets=TARGETS, adjacency="both", out_targets=None, rounds=ROUNDS):
    """All trajectories of one mechanism. ``rounds`` other than ``ROUNDS`` is for smoke tests."""
    center, rho, quantile, z = CONFIGS[config]
    combos = [combo for seed in seeds for combo in build_combos(
        alpha=ALPHA, seeds=[seed], targets=list(targets), clients=SLOTS, privacy="global-dp", ratios=[RATIO],
        adjacency=adjacency, out_targets=out_targets, diagnostics=True,
        partition_seed=None if seed == LAYOUT_SEED else LAYOUT_SEED)]
    return [ResidualCombo(**{
        **{field.name: getattr(combo, field.name) for field in fields(FrontierCombo)},
        "name_prefix": f"{config}-{combo.name_prefix}",
        "privacy": "residual-clipping", "noise_multiplier": z, "aggregation_weighting": "equal",
        "hyperparams": replace(HYPERPARAMS, rounds=rounds),
        "config": config, "center": center, "rho": rho, "quantile": quantile,
    }) for combo in combos]


def pinned_config(combo):
    return {"schema": CONFIG_SCHEMA, "config": combo.config, "center": combo.center, "rho": combo.rho,
            "quantile": combo.quantile, "noise_multiplier": combo.noise_multiplier, "slots": SLOTS,
            "clipping_norm": CLIP, "bit_noise_std": BIT_NOISE_STD if combo.quantile is not None else None,
            "rho_bounds": list(RHO_BOUNDS) if combo.quantile is not None else None,
            "mu_per_round": mu_per_round(combo.config), "run_name": combo.run_name()}


def plan(configs, seeds=SEEDS, adjacency="both", out_targets=None, rounds=ROUNDS):
    return [(Path(config), build_residual_combos(config, seeds, TARGETS, adjacency, out_targets, rounds))
            for config in configs]


def _run_isolated(combo, targets, output, max_parallel_clients):
    """Child-process entry point for one trajectory."""
    from results.cia_frontier.eurosat_frontier.runner import _execute

    folder = output / combo.run_name()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "residual_config.json"
    pinned = pinned_config(combo)
    if path.exists() and json.loads(path.read_text()) != pinned:
        raise ValueError(f"Configuration mismatch: {folder}; use a separate output directory")
    atomic_json(path, pinned)
    _execute([combo], targets, output, max_parallel_clients)


def main():
    from results.cia_frontier.signature_suppression.runner import execute_shared

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                        help="42 is the protocol; 43 only as the declared extension")
    parser.add_argument("--adjacency", choices=("both", "in", "out"), default="both")
    parser.add_argument("--out-targets", type=int, nargs="+")
    parser.add_argument("--smoke-rounds", type=int, help="smoke test only: at most 5 rounds, --output outside results/")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=8)
    parser.add_argument("--stall-minutes", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rounds, output = ROUNDS, args.output or DEFAULT_OUTPUT
    if args.smoke_rounds is not None:
        if not 1 <= args.smoke_rounds <= 5:
            parser.error("--smoke-rounds must be between 1 and 5")
        if args.output is None or (PROJECT_ROOT / "results") in args.output.resolve().parents:
            parser.error("--smoke-rounds requires an --output outside results/")
        rounds = args.smoke_rounds
    cells = plan(args.configs, args.seeds, args.adjacency, args.out_targets, rounds)
    combos = [combo for _, cell in cells for combo in cell]
    complete = sum((output / sub / c.run_name() / "complete.json").exists() for sub, cell in cells for c in cell)
    print(json.dumps({
        "trajectories": len(combos), "complete": complete, "execute": args.execute, "rounds": rounds,
        "targets": list(TARGETS), "seeds": args.seeds,
        "trajectory_hours_planned": round(len(combos) * rounds * TRAJECTORY_HOURS_PER_ROUND, 2),
        "configs": {str(sub): pinned_config(cell[0]) for sub, cell in cells},
    }, indent=2))
    if args.execute:
        for sub, cell in cells:
            execute_shared(cell, list(TARGETS), output / sub, args.max_parallel_clients,
                           stall_minutes=args.stall_minutes, worker=_run_isolated)


if __name__ == "__main__":
    main()
