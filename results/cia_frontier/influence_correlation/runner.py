"""influence_correlation: which influence definition predicts a client's attack score?

Small federations so EVERY client gets an attack score: 16 clients x 450 images (a 7,200-image
stratified pool of the Stage B datasets), cifar10_cnn, Dirichlet a=0.3, 50 rounds. Per
(dataset, arm, seed): one IN trajectory scoring all 16 clients every round and logging per-client
influence, plus 16 OUT trajectories (one client removed each). Arms: vanilla, global-dp and
metric-privacy at one ratio in the gap region (0.004).

Everything else (profiles, shadow sets, influence log, measurement format) is Stage B's,
through dataset_vs_model.stage_b, so per_client_score.py and the Stage C tooling apply unchanged.

Plans by default; training requires --execute. Never launches remote workers itself.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from results.cia_frontier.dataset_vs_model import stage_b

CLIENTS = 16
IMAGES_PER_CLIENT = 450
POOL = CLIENTS * IMAGES_PER_CLIENT  # 7,200, same data per client as Stage B/C
ROUNDS = 50
RATIO = 0.004
DATASETS = ("cifar10s", "eurosat32")
PRIVACY = stage_b.PRIVACY
SEEDS = (42,)  # seed 43 only after seed 42 is reviewed (PLAN.md)
TARGETS = tuple(range(CLIENTS))
DEFAULT_OUTPUT = Path("results/cia_frontier/influence_correlation/results")


def build_combos(*, seeds=SEEDS, datasets=DATASETS, privacy=PRIVACY, ratio=RATIO,
                 rounds=ROUNDS, clients=CLIENTS, out_targets=None, with_in=True):
    """IN trajectories (all clients as targets) and/or OUT trajectories for every client."""
    bad = set(datasets) - set(DATASETS)
    if bad or len(set(datasets)) != len(datasets):
        raise ValueError(f"Unknown or repeated datasets: {sorted(bad) or datasets}")
    if clients < 2 or clients * IMAGES_PER_CLIENT > 21_600:
        raise ValueError("clients must be >= 2 and fit the 21,600-image Stage B pools")
    targets = tuple(range(clients))
    common = dict(seeds=seeds, cells=[stage_b.cell_name(d, "dirichlet") for d in datasets],
                  ratios=[ratio], privacy=privacy, rounds=rounds, clients=clients,
                  targets=targets, train_subsample=clients * IMAGES_PER_CLIENT, stage="c")
    combos = stage_b.build_combos(**common) if with_in else []
    if out_targets is not None:
        combos += stage_b.build_combos(**common, out_targets=list(out_targets))
    return [replace(c, name_prefix=c.name_prefix.replace("stage-c-", "infl-", 1)) for c in combos]


def _record_dataset(dataset):
    """Walk Subset/Noisy wrappers down to the RecordImageDataset."""
    from torch.utils.data import Subset
    from metricdp_pytorch.utils.noisy_dataset import NoisyDataset
    while isinstance(dataset, (Subset, NoisyDataset)):
        dataset = dataset.dataset
    return dataset


def partition_summary(combo, base) -> dict:
    """Stage B's summary plus each client's class counts (labels read without decoding)."""
    from results.cia_frontier.dataset_vs_model.stage_b import root_indices
    summary = stage_b.partition_summary(combo, base)
    for client in summary["clients"]:
        train, _ = base.client_loaders(client["target"], num_partitions=combo.canonical_clients,
                                       partition_mode=combo.partition, batch_size=32,
                                       seed=combo.seed, dirichlet_alpha=combo.dirichlet_alpha)
        records = _record_dataset(train.dataset)
        labels = [int(records.dataset[i][records.label_column]) for i in root_indices(train.dataset)]
        client["class_counts"] = {str(k): labels.count(k) for k in sorted(set(labels))}
    return summary


def execute(combos, targets, output: Path, max_parallel_clients: int):
    """stage_b.execute with the class-count partition summary."""
    from experiments.cia.trajectories import execute_locked, execute_trajectory

    def run(combo):
        chosen = list(targets) if combo.out_target is None else [combo.out_target]
        base = stage_b._base(combo)
        execute_trajectory(
            combo, chosen=chosen, manifest=stage_b.manifest(combo, targets), output=output,
            max_parallel_clients=max_parallel_clients,
            partition_summary=lambda: partition_summary(combo, base),
            make_shadows=lambda target: stage_b.shadow_modules(combo, target, base),
            eval_seed=combo.seed,
            shadow_fingerprints=lambda shadows: stage_b.shadow_fingerprints(combo, shadows),
        )
    execute_locked(combos, run, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--privacy", nargs="+", choices=PRIVACY, default=list(PRIVACY))
    parser.add_argument("--ratio", type=float, default=RATIO)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--out-targets", type=int, nargs="+",
                        help="Train these OUT trajectories (default with --all-out: every client)")
    parser.add_argument("--all-out", action="store_true", help="OUT trajectories for all clients")
    parser.add_argument("--no-in", action="store_true", help="Skip the IN trajectories")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=6)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_parallel_clients < 1:
        parser.error("--max-parallel-clients must be positive")
    if args.all_out and args.out_targets:
        parser.error("Use either --all-out or --out-targets")
    out_targets = list(TARGETS) if args.all_out else args.out_targets
    combos = build_combos(seeds=args.seeds, datasets=args.datasets, privacy=args.privacy,
                          ratio=args.ratio, rounds=args.rounds, out_targets=out_targets,
                          with_in=not args.no_in)
    print(json.dumps({"training_runs": len(combos), "execute": args.execute,
                      "clients": CLIENTS, "pool": POOL, "out_targets": out_targets,
                      "run_names": [c.run_name() for c in combos]}, indent=2), flush=True)
    if args.execute:
        execute(combos, TARGETS, args.output.resolve(), args.max_parallel_clients)


if __name__ == "__main__":
    main()
