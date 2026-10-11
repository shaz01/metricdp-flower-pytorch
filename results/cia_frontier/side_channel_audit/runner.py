"""WP1 (D2 Stage A) side-channel and increment-attacker audit. Plans unless --execute.

Design (PROTOCOL.md): EuroSAT and Alzheimer MRI, label-Dirichlet alpha 0.3, 3, 5 and 8
canonical clients, every client a target (removal adjacency: one IN trajectory per seed,
one OUT trajectory per target and seed), global DP and metric privacy with a noise
standard deviation that does not depend on the active client count (noise multiplier =
ratio x active clients, the frontier convention), equal aggregation weights, seeds 42-44,
50 rounds, WP0 diagnostics on (round-0 checkpoint, shadow increments and gradient norms,
released-update norms, server-side leave-one-out residual norms).

Several workers may run the same command (e.g. one per GPU): every trajectory is locked
while it trains, a locked trajectory is skipped, and completed ones are skipped on restart.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from results.cia_frontier.eurosat_frontier.runner import EUROSAT_DATA_MODULE, build_combos
from results.contest_at_scale.auc_frontier.alzheimer_remove import HYPERPARAMS as ALZHEIMER_HYPERPARAMS
from results.contest_at_scale.auc_frontier.eurosat_remove import HYPERPARAMS as EUROSAT_HYPERPARAMS

ALPHA = 0.3
CLIENT_COUNTS = (3, 5, 8)
MECHANISMS = ("global-dp", "metric-privacy")
SEEDS = (42, 43, 44)
ROUNDS = 50
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "cia_frontier" / "side_channel_audit" / "results"

# Noise ratio r: sigma = r * C for global DP whatever the active count; metric privacy
# divides by the round's max pairwise distance. EuroSAT: the committed frontier noise.
# Alzheimer: z = 0.01 at 3 clients (the plan-suite runs behind evidence E2), r = 0.01 / 3.
# Cost: training wall time relative to EuroSAT, measured on the owner PC (3 rounds,
# 3 and 8 clients: 67 s vs 12-15 s per round).
DATASETS = {
    "eurosat": dict(
        data_module=EUROSAT_DATA_MODULE,
        model_module="experiments.reproduce.eurosat_cnn:create_model",
        hyperparams=replace(EUROSAT_HYPERPARAMS, rounds=ROUNDS),
        ratio=0.001546,
        relative_cost=1.0,
    ),
    "alzheimer": dict(
        data_module="results.cia_frontier.side_channel_audit.data:create_data_module",
        model_module="experiments.reproduce.paper_cnn:create_model",
        hyperparams=replace(ALZHEIMER_HYPERPARAMS, rounds=ROUNDS),
        ratio=0.01 / 3,
        relative_cost=0.2,
    ),
}
# Roadmap unit: one 48-client, 100-round EuroSAT trajectory on the reference machine
# (0.68 h training + 0.06 h evaluation).
TRAJECTORY_HOURS_PER_ROUND = 0.74 / 100


def build_audit_combos(dataset, clients, privacy, seeds=SEEDS, adjacency="both", out_targets=None,
                       rounds=ROUNDS):
    """All trajectories of one (dataset, client count, mechanism) cell; every client is a target.

    ``rounds`` other than ``ROUNDS`` is for smoke tests only (run names and manifests differ).
    """
    spec = DATASETS[dataset]
    combos = build_combos(alpha=ALPHA, seeds=list(seeds), targets=list(range(clients)),
                          clients=clients, privacy=privacy, ratios=[spec["ratio"]],
                          adjacency=adjacency, out_targets=out_targets, diagnostics=True)
    return [replace(
        combo,
        name_prefix=dataset + combo.name_prefix.removeprefix("eurosat"),
        data_module=spec["data_module"], model_module=spec["model_module"],
        hyperparams=replace(spec["hyperparams"], rounds=rounds), aggregation_weighting="equal",
    ) for combo in combos]


def plan(datasets, client_counts, mechanisms, seeds, adjacency="both", out_targets=None, rounds=ROUNDS):
    """``[(output subdirectory, targets, combos)]`` for every selected cell."""
    cells = []
    for dataset in datasets:
        for clients in client_counts:
            for privacy in mechanisms:
                combos = build_audit_combos(dataset, clients, privacy, seeds, adjacency, out_targets, rounds)
                cells.append((Path(dataset) / f"clients-{clients}" / privacy, list(range(clients)), combos))
    return cells


def trajectory_hours(combo, dataset):
    return combo.hyperparams.rounds * TRAJECTORY_HOURS_PER_ROUND * DATASETS[dataset]["relative_cost"]


def execute_shared(combos, targets, output, max_parallel_clients):
    """Train and evaluate each trajectory unless another worker holds it."""
    import fcntl
    import hashlib
    import tempfile

    from results.cia_frontier.eurosat_frontier.runner import _execute

    for combo in combos:
        identity = str((output / combo.run_name()).resolve()).encode()
        lock = Path(tempfile.gettempdir()) / ("auc-frontier-" + hashlib.sha256(identity).hexdigest() + ".lock")
        with lock.open("w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"[AUDIT] skip (another worker holds it): {combo.run_name()}", flush=True)
                continue
            _execute([combo], targets, output, max_parallel_clients)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--clients", type=int, nargs="+", choices=CLIENT_COUNTS, default=list(CLIENT_COUNTS))
    parser.add_argument("--privacy", nargs="+", choices=MECHANISMS, default=list(MECHANISMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS),
                        help="42-44 are the protocol; 45 only as the declared extension")
    parser.add_argument("--adjacency", choices=("both", "in", "out"), default="both")
    parser.add_argument("--out-targets", type=int, nargs="+")
    parser.add_argument("--smoke-rounds", type=int,
                        help="smoke test only: at most 5 rounds, requires an --output outside results/")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_parallel_clients < 1:
        parser.error("--max-parallel-clients must be positive")
    rounds = ROUNDS
    if args.smoke_rounds is not None:
        if not 1 <= args.smoke_rounds <= 5:
            parser.error("--smoke-rounds must be between 1 and 5")
        if args.output is None or args.output.resolve().is_relative_to(PROJECT_ROOT / "results"):
            parser.error("--smoke-rounds needs an --output outside results/")
        rounds = args.smoke_rounds
    output = args.output or DEFAULT_OUTPUT
    cells = plan(args.datasets, args.clients, args.privacy, args.seeds, args.adjacency, args.out_targets, rounds)
    summary = [dict(cell=str(path), trajectories=len(combos),
                    trajectory_hours=round(sum(trajectory_hours(c, str(path).split("/")[0]) for c in combos), 2))
               for path, _, combos in cells]
    print(json.dumps({"execute": args.execute, "rounds": rounds, "seeds": args.seeds,
                      "trajectories": sum(c["trajectories"] for c in summary),
                      "trajectory_hours": round(sum(c["trajectory_hours"] for c in summary), 2),
                      "cells": summary}, indent=2))
    if args.execute:
        for path, targets, combos in cells:
            execute_shared(combos, targets, output / path, args.max_parallel_clients)


if __name__ == "__main__":
    main()
