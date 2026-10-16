"""Step 3: consensus KL 0.3 under global DP against global DP alone. Plans unless --execute.

Design (PROTOCOL.md, step 3): WP2's federation (EuroSAT, alpha 0.3, 48 clients, 50 rounds, equal weights, targets 23,
7, 1, 0, 3, WP0 diagnostics) with server-side global DP at noise ratios r in {1, 2, 4} x 0.001546 (noise std r C,
fixed across removal by the frontier convention: noise multiplier r x active clients), with and without consensus
KL beta 0.3. Uses WP2's training option and stall-guarded executor unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path

from results.cia_frontier.eurosat_frontier.runner import FrontierCombo, build_combos
from results.cia_frontier.signature_suppression.runner import (
    ALPHA, CLIENTS, HYPERPARAMS, LAYOUT_SEED, TARGETS, SuppressionCombo, _run_isolated, execute_shared, pinned_config,
)

BASE_RATIO = 0.001546
RATIOS = (BASE_RATIO, 2 * BASE_RATIO, 4 * BASE_RATIO)
BETAS = {"gdp": 0.0, "gdp-consensus-kl-0.3": 0.3}
SEEDS = (42,)
ROUNDS = 50
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "cia_frontier" / "consensus_kl" / "results" / "step3"
TRAJECTORY_HOURS_PER_ROUND = 0.74 / 100


def cell_name(mechanism, ratio):
    return f"{mechanism}-r{ratio!r}"


def build_dp_combos(mechanism, ratio, seeds=SEEDS, adjacency="both", out_targets=None, rounds=ROUNDS):
    combos = [combo for seed in seeds for combo in build_combos(
        alpha=ALPHA, seeds=[seed], targets=list(TARGETS), clients=CLIENTS, privacy="global-dp", ratios=[ratio],
        adjacency=adjacency, out_targets=out_targets, diagnostics=True,
        partition_seed=None if seed == LAYOUT_SEED else LAYOUT_SEED)]
    beta = BETAS[mechanism]
    return [SuppressionCombo(**{
        **{field.name: getattr(combo, field.name) for field in fields(FrontierCombo)},
        "name_prefix": f"{mechanism}-{combo.name_prefix}",
        "aggregation_weighting": "equal", "hyperparams": replace(HYPERPARAMS, rounds=rounds),
        "config": cell_name(mechanism, ratio), "consensus_beta": beta,
    }) for combo in combos]


def plan(mechanisms, ratios, seeds=SEEDS, adjacency="both", out_targets=None, rounds=ROUNDS):
    return [(Path(cell_name(m, r)), build_dp_combos(m, r, seeds, adjacency, out_targets, rounds))
            for m in mechanisms for r in ratios]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanisms", nargs="+", choices=tuple(BETAS), default=list(BETAS))
    parser.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--adjacency", choices=("both", "in", "out"), default="both")
    parser.add_argument("--out-targets", type=int, nargs="+")
    parser.add_argument("--smoke-rounds", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=8)
    parser.add_argument("--stall-minutes", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if any(r not in RATIOS for r in args.ratios):
        parser.error(f"ratios must come from the protocol grid {RATIOS}")
    rounds, output = ROUNDS, args.output or DEFAULT_OUTPUT
    if args.smoke_rounds is not None:
        if not 1 <= args.smoke_rounds <= 5:
            parser.error("--smoke-rounds must be between 1 and 5")
        if args.output is None or (PROJECT_ROOT / "results") in args.output.resolve().parents:
            parser.error("--smoke-rounds requires an --output outside results/")
        rounds = args.smoke_rounds
    cells = plan(args.mechanisms, args.ratios, args.seeds, args.adjacency, args.out_targets, rounds)
    combos = [c for _, cell in cells for c in cell]
    complete = sum((output / sub / c.run_name() / "complete.json").exists() for sub, cell in cells for c in cell)
    print(json.dumps({"trajectories": len(combos), "complete": complete, "execute": args.execute, "rounds": rounds,
                      "trajectory_hours_planned": round(len(combos) * rounds * TRAJECTORY_HOURS_PER_ROUND, 2),
                      "cells": {str(sub): pinned_config(cell[0]) | {"noise_ratio": cell[0].noise_ratio,
                                                                     "noise_multiplier": cell[0].noise_multiplier}
                                for sub, cell in cells}}, indent=2))
    if args.execute:
        for sub, cell in cells:
            execute_shared(cell, list(TARGETS), output / sub, args.max_parallel_clients,
                           stall_minutes=args.stall_minutes, worker=_run_isolated)


if __name__ == "__main__":
    main()
