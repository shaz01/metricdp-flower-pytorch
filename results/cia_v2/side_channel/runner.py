"""A2: metric privacy's noise-level side channel at strong noise, small federations (CIA roadmap v2). Plans unless --execute.

WP1's small-federation trajectories (``results/cia_frontier/side_channel_audit/runner.py``: label-Dirichlet alpha 0.3,
equal weights, 50 rounds, WP0 diagnostics, noise std independent of the active count) with three changes: the layout
is fixed at seed 42 (``partition_seed``), only the largest client is a target, and every trajectory draws its own
client and noise seeds from ``SeedSequence([3, cell, world, replicate])`` (pilot replicates from a disjoint range).
Replicate ``j`` of IN and OUT are listed as a pair (IN first), the column layout of the pair jackknife.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields, replace
import json
from pathlib import Path

import numpy as np

from results.cia_frontier.eurosat_frontier.runner import FrontierCombo, atomic_json, build_combos
from results.cia_frontier.side_channel_audit.runner import DATASETS, _run_isolated, execute_shared

ALPHA = 0.3
LAYOUT_SEED = 42
ROUNDS = 50
# (dataset, clients) -> largest client of the seed-42 layout (STATE.md, Design).
CELLS = {("alzheimer", 3): 2, ("alzheimer", 5): 1, ("alzheimer", 8): 5, ("eurosat", 3): 1}
BASE_RATIO = {name: spec["ratio"] for name, spec in DATASETS.items()}  # WP1: EuroSAT 0.001546, Alzheimer 0.01/3
MULTIPLES = (1, 4, 16, 64)
MECHANISMS = ("global-dp", "metric-privacy")
REPLICATES = {"pilot": 4, "main": 8}
STAGE_OFFSET = {"pilot": 1000, "main": 0}  # pilot seeds never coincide with main-run seeds
SEED_ENTROPY = 3
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def trajectory_seeds(dataset, clients, world, replicate, stage):
    cell = list(CELLS).index((dataset, clients))
    state = np.random.SeedSequence([SEED_ENTROPY, cell, int(world), STAGE_OFFSET[stage] + replicate]).generate_state(2)
    return int(state[0]), int(state[1])


@dataclass(frozen=True, kw_only=True)
class SideChannelCombo(FrontierCombo):
    replicate: int
    stage: str
    client_seed: int
    noise_seed: int

    def runner_args(self, **kwargs):
        return (*super().runner_args(**kwargs), "--client-seed", str(self.client_seed),
                "--noise-seed", str(self.noise_seed), "--log-noise-diagnostics")


def build_cell(dataset, clients, privacy, *, multiple, stage, rounds=ROUNDS):
    """IN and OUT-target trajectories of one cell, mechanism and noise multiple, paired by replicate."""
    if multiple not in MULTIPLES:
        raise ValueError(f"noise multiples come from the protocol grid {MULTIPLES}")
    if privacy not in MECHANISMS:
        raise ValueError(f"privacy must be one of {MECHANISMS}")
    target = CELLS[(dataset, clients)]
    spec = DATASETS[dataset]
    ratio = BASE_RATIO[dataset] * multiple
    base = build_combos(alpha=ALPHA, seeds=[LAYOUT_SEED], targets=[target], clients=clients, privacy=privacy,
                        ratios=[ratio], partition_seed=LAYOUT_SEED, diagnostics=True)
    worlds = {combo.out_target: combo for combo in base}
    combos = []
    for j in range(REPLICATES[stage]):
        for world, out_target in enumerate((None, target)):
            combo = worlds[out_target]
            client_seed, noise_seed = trajectory_seeds(dataset, clients, world, j, stage)
            values = {f.name: getattr(combo, f.name) for f in fields(FrontierCombo)}
            values.update(
                name_prefix=f"a2-{stage}-x{multiple}-rep{j:02d}-{dataset}" + combo.name_prefix.removeprefix("eurosat"),
                data_module=spec["data_module"], model_module=spec["model_module"],
                hyperparams=replace(spec["hyperparams"], rounds=rounds), aggregation_weighting="equal",
            )
            combos.append(SideChannelCombo(**values, replicate=j, stage=stage, client_seed=client_seed,
                                           noise_seed=noise_seed))
    return combos


def cell_directory(output, dataset, clients, privacy, multiple, stage):
    return output / stage / dataset / f"clients-{clients}" / f"{privacy}-x{multiple}"


def _worker(combo, targets, output, max_parallel_clients):
    """Record the trajectory's seeds, then run WP1's trajectory worker unchanged."""
    folder = output / combo.run_name()
    folder.mkdir(parents=True, exist_ok=True)
    seeds = {"stage": combo.stage, "replicate": combo.replicate,
             "world": "IN" if combo.out_target is None else f"OUT-{combo.out_target}",
             "client_seed": combo.client_seed, "noise_seed": combo.noise_seed, "seed_entropy": SEED_ENTROPY}
    path = folder / "a2_seeds.json"
    if path.exists() and json.loads(path.read_text()) != seeds:
        raise ValueError(f"Seed mismatch: {folder}; use a separate output directory")
    atomic_json(path, seeds)
    _run_isolated(combo, targets, output, max_parallel_clients)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="+", required=True, help="dataset:clients, e.g. alzheimer:3 eurosat:3")
    parser.add_argument("--mechanisms", nargs="+", choices=MECHANISMS, required=True)
    parser.add_argument("--multiple", type=int, choices=MULTIPLES, required=True)
    parser.add_argument("--stage", choices=tuple(REPLICATES), required=True)
    parser.add_argument("--smoke-rounds", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=4)
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
    cells = []
    for item in args.cells:
        dataset, clients = item.split(":")
        for privacy in args.mechanisms:
            combos = build_cell(dataset, int(clients), privacy, multiple=args.multiple, stage=args.stage,
                                rounds=rounds)
            cells.append((cell_directory(output, dataset, int(clients), privacy, args.multiple, args.stage),
                          [CELLS[(dataset, int(clients))]], combos))
    print(json.dumps({"trajectories": sum(len(c) for _, _, c in cells), "execute": args.execute, "rounds": rounds,
                      "cells": [str(d) for d, _, _ in cells]}, indent=2))
    if args.execute:
        for directory, targets, combos in cells:
            execute_shared(combos, targets, directory, args.max_parallel_clients,
                           stall_minutes=args.stall_minutes, worker=_worker)


if __name__ == "__main__":
    main()
