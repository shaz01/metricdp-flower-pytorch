"""Stage B of dataset_vs_model: how do the dataset and label skew change the accuracy gap?

One model (cifar10_cnn, the one where the two datasets behave differently in Stage A).
Cells: {cifar10s, eurosat32} x {homogeneous, dirichlet a=0.3} x {vanilla, global-dp x 4
ratios, metric-privacy x 4 ratios} x seeds. Both datasets have a 21,600-image pool, so
clients get the same amount of data. Accuracy only; the per-round DP diagnostics
(update norms, client distances, noise) are logged by the strategies as in Stage A.

Plans by default; training requires --execute. Never launches remote workers itself.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

from experiments.reproduce.matrix import Combo, run_combos
from results.cia_frontier.dataset_vs_model.data import STAGE_B_DATASETS
from results.cia_frontier.dataset_vs_model.runner import HYPERPARAMS as _STAGE_A_HYPERPARAMS, MODELS

ROUNDS = 20
CLIENTS = 48
MODEL = "cifar10_cnn"
DIRICHLET_ALPHA = 0.3
RATIOS = (0.001, 0.0025, 0.004, 0.00625)  # contest_at_scale/cifar10's three, plus one below
PRIVACY = ("vanilla", "global-dp", "metric-privacy")
PARTITIONS = ("homogeneous", "dirichlet")
SEEDS = (42, 43, 44)
HYPERPARAMS = replace(_STAGE_A_HYPERPARAMS, rounds=ROUNDS)
DEFAULT_OUTPUT = Path("results/cia_frontier/dataset_vs_model/results/stage_b")

CELLS = tuple((d, p) for d in STAGE_B_DATASETS for p in PARTITIONS)


def cell_name(dataset: str, partition: str) -> str:
    return f"{dataset}+{partition}"


def arm_name(privacy: str, ratio: float | None) -> str:
    return "vanilla" if privacy == "vanilla" else f"{privacy}-r{ratio!r}"


def build_combos(*, seeds=SEEDS, cells=None, ratios=RATIOS, privacy=PRIVACY,
                 rounds=ROUNDS, clients=CLIENTS, alpha=DIRICHLET_ALPHA):
    """One accuracy-only run per (cell, arm, seed). ``cells`` filters for sharding."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seeds")
    if not ratios or any(not math.isfinite(r) or r <= 0 for r in ratios) or len(set(ratios)) != len(ratios):
        raise ValueError("Ratios must be distinct, finite and positive")
    if clients < 2 or rounds < 1 or alpha <= 0:
        raise ValueError("Require clients >= 2, rounds >= 1, alpha > 0")
    unknown = set(privacy) - set(PRIVACY)
    if unknown:
        raise ValueError(f"Unknown privacy modes: {sorted(unknown)}")
    wanted = [cell_name(*c) for c in CELLS] if cells is None else list(cells)
    bad = set(wanted) - {cell_name(*c) for c in CELLS}
    if bad or len(set(wanted)) != len(wanted):
        raise ValueError(f"Unknown or repeated cells: {sorted(bad) or wanted}")
    arms = [(p, None) if p == "vanilla" else (p, r) for p in privacy for r in ([None] if p == "vanilla" else ratios)]
    return [Combo(
        name_prefix=f"stage-b-{dataset}-{partition}-{arm_name(priv, ratio)}",
        num_clients=clients, partition=partition, dirichlet_alpha=alpha,
        privacy=priv, aggregation="fedavg", seed=seed,
        noise_multiplier=0.0 if ratio is None else ratio * clients,
        hyperparams=replace(HYPERPARAMS, rounds=rounds),
        data_module=STAGE_B_DATASETS[dataset], model_module=MODELS[MODEL], data_tag=dataset,
    ) for dataset, partition in CELLS if cell_name(dataset, partition) in wanted
      for priv, ratio in arms for seed in seeds]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--cells", nargs="+", choices=[cell_name(*c) for c in CELLS],
                        help="Subset of cells to train (sharding across machines)")
    parser.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    parser.add_argument("--privacy", nargs="+", choices=PRIVACY, default=list(PRIVACY))
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=6)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_parallel_clients < 1:
        parser.error("--max-parallel-clients must be positive")
    combos = build_combos(seeds=args.seeds, cells=args.cells, ratios=args.ratios,
                          privacy=args.privacy, rounds=args.rounds)
    print(json.dumps({"training_runs": len(combos), "execute": args.execute,
                      "run_names": [c.run_name() for c in combos]}, indent=2), flush=True)
    if args.execute:
        output = args.output.resolve()
        run_combos(combos, output_dir=output, max_parallel_clients=args.max_parallel_clients)
        missing = [c.run_name() for c in combos if not c.result_path(output).exists()]
        if missing:
            raise SystemExit(f"{len(missing)} run(s) did not finish: {missing}")


if __name__ == "__main__":
    main()
