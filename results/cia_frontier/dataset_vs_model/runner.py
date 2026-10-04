"""Stage A of dataset_vs_model: is the CIFAR-10 noise gap caused by the dataset or the model?

2x2 cells (dataset x model), homogeneous, 48 clients, metric-privacy only, 10 rounds.
Each run already logs per-round client update norms before clipping and all pairwise
client-model distances (see analyze.py), so accuracy-only training is enough.

Plans by default; training requires --execute. Never launches remote workers itself.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

from experiments.reproduce.matrix import Combo, run_combos
from results.cia_frontier.dataset_vs_model.data import DATASETS
from results.contest_at_scale.auc_frontier.eurosat_remove import HYPERPARAMS as _FRONTIER_HYPERPARAMS

ROUNDS = 10
CLIENTS = 48
NOISE_RATIO = 0.0025  # noise multiplier = ratio x clients, the PLAN.md convention
HYPERPARAMS = replace(_FRONTIER_HYPERPARAMS, rounds=ROUNDS)

MODELS = {
    "eurosat_cnn": "experiments.reproduce.eurosat_cnn:create_model",
    "cifar10_cnn": "experiments.reproduce.cifar10_cnn:create_model",
}
CELLS = tuple((dataset, model) for dataset in DATASETS for model in MODELS)
DEFAULT_OUTPUT = Path("results/cia_frontier/dataset_vs_model/results/stage_a")


def cell_name(dataset: str, model: str) -> str:
    return f"{dataset}+{model}"


def build_combos(*, seeds, cells=None, ratio=NOISE_RATIO, clients=CLIENTS,
                 privacy="metric-privacy", rounds=ROUNDS):
    """One accuracy-only run per (cell, seed). ``cells`` filters for sharding."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seeds")
    if not math.isfinite(ratio) or ratio <= 0 or clients < 2 or rounds < 1:
        raise ValueError("Require finite ratio > 0, clients >= 2, rounds >= 1")
    if privacy not in ("vanilla", "global-dp", "metric-privacy"):
        raise ValueError("Unknown privacy mode")
    wanted = [cell_name(*c) for c in CELLS] if cells is None else list(cells)
    unknown = set(wanted) - {cell_name(*c) for c in CELLS}
    if unknown or len(set(wanted)) != len(wanted):
        raise ValueError(f"Unknown or repeated cells: {sorted(unknown) or wanted}")
    return [Combo(
        name_prefix=f"stage-a-{dataset}-{model}-r{ratio!r}",
        num_clients=clients, partition="homogeneous", privacy=privacy, aggregation="fedavg",
        seed=seed, noise_multiplier=0.0 if privacy == "vanilla" else ratio * clients,
        hyperparams=replace(HYPERPARAMS, rounds=rounds),
        data_module=DATASETS[dataset], model_module=MODELS[model], data_tag=dataset,
    ) for dataset, model in CELLS if cell_name(dataset, model) in wanted for seed in seeds]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--cells", nargs="+", choices=[cell_name(*c) for c in CELLS],
                        help="Subset of cells to train (sharding across machines)")
    parser.add_argument("--ratio", type=float, default=NOISE_RATIO)
    parser.add_argument("--privacy", choices=("vanilla", "global-dp", "metric-privacy"),
                        default="metric-privacy")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=6)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_parallel_clients < 1:
        parser.error("--max-parallel-clients must be positive")
    combos = build_combos(seeds=args.seeds, cells=args.cells, ratio=args.ratio,
                          privacy=args.privacy, rounds=args.rounds)
    print(json.dumps({"training_runs": len(combos), "execute": args.execute,
                      "noise_multiplier": combos[0].noise_multiplier if combos else None,
                      "run_names": [c.run_name() for c in combos]}, indent=2), flush=True)
    if args.execute:
        output = args.output.resolve()
        run_combos(combos, output_dir=output, max_parallel_clients=args.max_parallel_clients)
        missing = [c.run_name() for c in combos
                   if not c.result_path(output).exists()]
        if missing:
            raise SystemExit(f"{len(missing)} run(s) did not finish: {missing}")


if __name__ == "__main__":
    main()
