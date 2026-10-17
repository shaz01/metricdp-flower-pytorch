"""M1 shared-noise confirmation (CIA roadmap v2). Plans unless --execute.

v1's consensus-KL step 3 global-DP arm (``results/cia_frontier/consensus_kl/runner_dp.py``, mechanism ``gdp``) at 1x
and 4x the frontier noise, unchanged except that every world (IN, OUT-23, OUT-7, OUT-1, OUT-0, OUT-3) draws its own
client and noise seeds, from ``SeedSequence([1, world])``. The same world reuses its seeds at both noise levels.
Public randomness (``seed`` 42: initial model, layout, test split, shadow samples) is shared as in v1.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import json
from pathlib import Path

import numpy as np

from results.cia_frontier.consensus_kl.runner_dp import BASE_RATIO, build_dp_combos, cell_name
from results.cia_frontier.signature_suppression.runner import (
    SuppressionCombo, _run_isolated, atomic_json, execute_shared,
)

RATIOS = (BASE_RATIO, 4 * BASE_RATIO)
WORLDS = (None, 23, 7, 1, 0, 3)  # IN, then OUT per target in v1's order
SEED_ENTROPY = 1
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"
TRAJECTORY_HOURS_PER_ROUND = 0.74 / 100


def world_seeds(world):
    client_seed, noise_seed = np.random.SeedSequence([SEED_ENTROPY, WORLDS.index(world)]).generate_state(2)
    return int(client_seed), int(noise_seed)


@dataclass(frozen=True, kw_only=True)
class DecoupledCombo(SuppressionCombo):
    client_seed: int
    noise_seed: int

    def runner_args(self, **kwargs):
        return (*super().runner_args(**kwargs), "--client-seed", str(self.client_seed),
                "--noise-seed", str(self.noise_seed), "--log-noise-diagnostics")


def plan(ratios=RATIOS, rounds=50):
    if any(r not in RATIOS for r in ratios):
        raise ValueError(f"ratios must come from the protocol grid {RATIOS}")
    cells = []
    for ratio in ratios:
        combos = []
        for combo in build_dp_combos("gdp", ratio, rounds=rounds):
            if combo.out_target not in WORLDS:
                raise ValueError(f"unexpected world {combo.out_target}")
            client_seed, noise_seed = world_seeds(combo.out_target)
            values = {field.name: getattr(combo, field.name) for field in fields(SuppressionCombo)}
            combos.append(DecoupledCombo(**values | {"name_prefix": "m1-" + combo.name_prefix},
                                         client_seed=client_seed, noise_seed=noise_seed))
        cells.append((cell_name("gdp", ratio), combos))
    return cells


def _worker(combo, targets, output, max_parallel_clients):
    """Record the world's seeds next to the trajectory, then run v1's trajectory worker unchanged."""
    folder = output / combo.run_name()
    folder.mkdir(parents=True, exist_ok=True)
    seeds = {"world": "IN" if combo.out_target is None else f"OUT-{combo.out_target}",
             "client_seed": combo.client_seed, "noise_seed": combo.noise_seed, "seed_entropy": SEED_ENTROPY}
    path = folder / "m1_seeds.json"
    if path.exists() and json.loads(path.read_text()) != seeds:
        raise ValueError(f"Seed mismatch: {folder}; use a separate output directory")
    atomic_json(path, seeds)
    _run_isolated(combo, targets, output, max_parallel_clients)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    parser.add_argument("--smoke-rounds", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=8)
    parser.add_argument("--stall-minutes", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rounds, output = 50, args.output or DEFAULT_OUTPUT
    if args.smoke_rounds is not None:
        if not 1 <= args.smoke_rounds <= 5:
            parser.error("--smoke-rounds must be between 1 and 5")
        if args.output is None or (PROJECT_ROOT / "results") in args.output.resolve().parents:
            parser.error("--smoke-rounds requires an --output outside results/")
        rounds = args.smoke_rounds
    cells = plan(args.ratios, rounds)
    combos = [c for _, cell in cells for c in cell]
    print(json.dumps({"trajectories": len(combos), "execute": args.execute, "rounds": rounds,
                      "trajectory_hours_planned": round(len(combos) * rounds * TRAJECTORY_HOURS_PER_ROUND, 2),
                      "complete": sum((output / sub / c.run_name() / "complete.json").exists()
                                      for sub, cell in cells for c in cell),
                      "worlds": [{"run_name": c.run_name(), "client_seed": c.client_seed, "noise_seed": c.noise_seed}
                                 for c in combos]}, indent=2))
    if args.execute:
        for sub, cell in cells:
            execute_shared(cell, list(WORLDS[1:]), output / sub, args.max_parallel_clients,
                           stall_minutes=args.stall_minutes, worker=_worker)


if __name__ == "__main__":
    main()
