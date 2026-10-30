"""Paper-protocol CIA runs: the published paper's 3-client Alzheimer setting (CIA roadmap v3, P0 item 4; used by P1).
Plans unless --execute.

The paper's multi-round CIA (Sáinz-Pardo Díaz et al. 2026, Section 7.4.2, Table 13): Alzheimer MRI split into three
clients by Table 9 (``experiments.cia.datasets.paper``, target client 3 = partition 2), FedAvg, 20 rounds, 5 local
epochs, batch 32, clipping norm 5, noise multiplier ``z`` with the paper's law (std ``z C / n_t`` for global DP,
divided by the round's maximum client distance for metric privacy). The attacker scores every round's global model on
a shadow sample: 10% of the target's training records with Gaussian pixel noise of 20% of the maximum value.

One pair is an IN world (3 clients) and an OUT world (the target removed, 2 clients). Unlike the paper and the plan
suite (``results/contest_at_scale/plan_suite``), each world draws its own ``client_seed`` and ``noise_seed`` from
``(design_seed, pair, world)``, so no two worlds share private randomness (measurement standard v2, item 1). Public
randomness is shared by every run: the data split, the initial model and the shadow samples (``layout-seed`` =
``init-seed`` = 42).

Each run stores, at rounds 0-20, the server test loss and accuracy and the target's clean and noisy shadow losses, so
P1 can compute both the paper's pooled-checkpoint ROC per pair and a calibrated per-trajectory attacker.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np

from experiments.cia.datasets.paper import PAPER_CIA_NUM_CLIENTS, PAPER_CIA_TARGET_PARTITION_ID
from experiments.reproduce.matrix import Combo
from results.contest_at_scale.plan_suite.planned_runs import HYPERPARAMS
from results.cia_v2.randomized_inclusion.runner import _loss_and_accuracy, atomic_json, check_every_client_trained

ROUNDS = 20
LAYOUT_SEED = 42
SHADOW_FRACTION = 0.10
SHADOW_NOISE_STD_FRACTION = 0.20
PRIVACY_MODES = ("vanilla", "global-dp", "metric-privacy")
WORLDS = ("in", "out")
DATA_MODULES = {"in": "results.contest_at_scale.plan_suite.planned_runs:create_alzheimer_in",
                "out": "results.contest_at_scale.plan_suite.planned_runs:create_alzheimer_out"}
MODEL_MODULE = "experiments.reproduce.paper_cnn:create_model"
TARGET = PAPER_CIA_TARGET_PARTITION_ID
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 1


def world_seeds(design_seed, pair, world):
    """``(client_seed, noise_seed)`` of one world: independent draws per (pair, world)."""
    client_seed, noise_seed = np.random.SeedSequence(
        [int(design_seed), int(pair), WORLDS.index(world)]).generate_state(2)
    return int(client_seed), int(noise_seed)


def checkpoint_rounds(rounds):
    """Every round, including the shared initial model (round 0); the paper's statistic uses rounds 1-20."""
    return list(range(rounds + 1))


@dataclass(frozen=True, kw_only=True)
class PaperRun(Combo):
    config: str
    pair: int
    pairs: int
    world: str
    design_seed: int
    client_seed: int
    noise_seed: int

    @property
    def active(self):
        return tuple(c for c in range(PAPER_CIA_NUM_CLIENTS) if self.world == "in" or c != TARGET)

    def runner_args(self, **kwargs):
        args = (*super().runner_args(**kwargs),
                "--layout-seed", str(LAYOUT_SEED), "--init-seed", str(LAYOUT_SEED),
                "--client-seed", str(self.client_seed), "--noise-seed", str(self.noise_seed),
                "--require-all-clients")
        return (*args, "--log-noise-diagnostics") if self.privacy != "vanilla" else args


def build_runs(*, config, privacy, noise_multiplier, pairs, design_seed, pair_subset=None, rounds=ROUNDS,
               aggregation_weighting="num-examples"):
    """Both worlds of every selected pair, IN first. ``noise_multiplier`` is the paper's ``z`` (std ``z C / n_t``)."""
    if privacy not in PRIVACY_MODES:
        raise ValueError(f"privacy must be one of {PRIVACY_MODES}")
    if privacy == "vanilla" and noise_multiplier:
        raise ValueError("vanilla takes no noise")
    if privacy != "vanilla" and not noise_multiplier > 0:
        raise ValueError("DP runs need a positive noise multiplier")
    selected = range(pairs) if pair_subset is None else sorted(set(pair_subset))
    runs = []
    for pair in selected:
        if not 0 <= pair < pairs:
            raise ValueError(f"pair {pair} outside the design")
        for world in WORLDS:
            client_seed, noise_seed = world_seeds(design_seed, pair, world)
            runs.append(PaperRun(
                name_prefix=f"paper-{config}-d{design_seed}-pair{pair:02d}-{world}",
                num_clients=PAPER_CIA_NUM_CLIENTS - (world == "out"), partition="non-iid", privacy=privacy,
                aggregation="fedavg", seed=LAYOUT_SEED, noise_multiplier=noise_multiplier,
                hyperparams=replace(HYPERPARAMS, rounds=rounds), data_module=DATA_MODULES[world],
                model_module=MODEL_MODULE, aggregation_weighting=aggregation_weighting, config=config, pair=pair,
                pairs=pairs, world=world, design_seed=design_seed, client_seed=client_seed, noise_seed=noise_seed,
            ))
    return runs


def manifest(run):
    return {"schema": SCHEMA, "run_name": run.run_name(), "config": run.config, "privacy": run.privacy,
            "noise_multiplier": run.noise_multiplier, "aggregation_weighting": run.aggregation_weighting,
            "pair": run.pair, "pairs": run.pairs, "world": run.world, "design_seed": run.design_seed,
            "active": list(run.active), "target": TARGET, "layout_seed": LAYOUT_SEED, "init_seed": LAYOUT_SEED,
            "client_seed": run.client_seed, "noise_seed": run.noise_seed,
            "checkpoint_rounds": checkpoint_rounds(run.hyperparams.rounds), "shadow_fraction": SHADOW_FRACTION,
            "shadow_noise_std_fraction": SHADOW_NOISE_STD_FRACTION, "hyperparams": asdict(run.hyperparams),
            "score_direction": "lower loss indicates IN"}


def evaluation_sets(batch_size):
    """Public test loader and the target's clean and noisy shadow loaders (identical in every run)."""
    from experiments.cia.datasets.paper import PaperCiaDataModule
    from experiments.cia.datasets.shadow import ShadowDataModule
    from metricdp_pytorch.utils.noisy_dataset import NoisyDataModule

    base = PaperCiaDataModule()
    _, test_loader = base.server_loaders(batch_size=batch_size, seed=LAYOUT_SEED)
    shadows = {}
    for view, module in (("shadow", base),
                         ("noisy_shadow", NoisyDataModule(base, std_fraction=SHADOW_NOISE_STD_FRACTION))):
        shadow = ShadowDataModule(module, num_clients=PAPER_CIA_NUM_CLIENTS, target_partition_id=TARGET,
                                  shadow_fraction=SHADOW_FRACTION, partition_mode="non-iid", partition_profile="auto")
        shadows[view] = shadow.target_shadow_loader(batch_size=batch_size, seed=LAYOUT_SEED)
    return test_loader, shadows


def execute_run(run, output, max_parallel_clients):
    """Train one world, score the target's shadow samples at every round, then delete the checkpoints."""
    import os
    import subprocess
    import sys
    import time

    import torch

    from experiments.cia.iter_combos import iter_combos
    from metricdp_pytorch.model_module import load_model
    from metricdp_pytorch.utils.device import resolve_device

    folder = output / run.run_name()
    folder.mkdir(parents=True, exist_ok=True)
    pinned = manifest(run)
    path = folder / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != pinned:
        raise ValueError(f"Manifest mismatch: {folder}; use a separate output directory")
    atomic_json(path, pinned)
    if (folder / "complete.json").exists():
        return
    revision = os.environ.get("METRICDP_SOURCE_COMMIT") or subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
    ).stdout.strip()
    device = resolve_device()
    atomic_json(folder / "provenance.json", {"commit": revision, "python": sys.version, "device": str(device)})
    rounds = tuple(pinned["checkpoint_rounds"])
    started = time.monotonic()
    for _, success, paths in iter_combos([run], output_dir=folder, max_parallel_clients=max_parallel_clients,
                                         force=True, log=print, checkpoint_rounds=rounds):
        if not success:
            raise RuntimeError(f"Training failed: {run.run_name()}")
        trained = time.monotonic()
        check_every_client_trained(folder / f"{run.run_name()}.json", len(run.active))
        test_loader, shadows = evaluation_sets(run.hyperparams.batch_size)
        rows = []
        for round_number, checkpoint in zip(rounds, paths, strict=True):
            model = load_model(run.model_module)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            row = {"round": round_number}
            row["test_loss"], row["test_accuracy"] = _loss_and_accuracy(model, test_loader, device)
            for view, loader in shadows.items():
                row[f"{view}_loss"], row[f"{view}_accuracy"] = _loss_and_accuracy(model, loader, device)
            rows.append(row)
            atomic_json(folder / "measurements.json", rows)
            checkpoint.unlink()
        atomic_json(folder / "complete.json", {"complete": True, "training_seconds": round(trained - started, 1),
                                               "evaluation_seconds": round(time.monotonic() - trained, 1)})


def _worker(run, targets, output, max_parallel_clients):
    """Child-process entry point for the shared stall-guarded executor (``targets`` unused)."""
    del targets
    execute_run(run, output, max_parallel_clients)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configuration name (part of run names)")
    parser.add_argument("--privacy", choices=PRIVACY_MODES, required=True)
    parser.add_argument("--noise-multiplier", type=float, default=0.0, help="the paper's z (DP only)")
    parser.add_argument("--pairs", type=int, required=True)
    parser.add_argument("--design-seed", type=int, required=True)
    parser.add_argument("--pair-subset", type=int, nargs="+", help="train only these pairs (sharding)")
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--aggregation-weighting", default="num-examples",
                        help="FedAvg client weights (the paper's Flower FedAvg weights by examples)")
    parser.add_argument("--smoke-rounds", type=int, help="1-5 rounds into an --output outside results/")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=3)
    parser.add_argument("--stall-minutes", type=float, default=30.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rounds = args.rounds
    if args.smoke_rounds is not None:
        if not 1 <= args.smoke_rounds <= 5:
            parser.error("--smoke-rounds must be between 1 and 5")
        if (PROJECT_ROOT / "results") in args.output.resolve().parents:
            parser.error("--smoke-rounds requires an --output outside results/")
        rounds = args.smoke_rounds
    runs = build_runs(config=args.config, privacy=args.privacy, noise_multiplier=args.noise_multiplier,
                      pairs=args.pairs, design_seed=args.design_seed, pair_subset=args.pair_subset, rounds=rounds,
                      aggregation_weighting=args.aggregation_weighting)
    print(json.dumps({"runs": len(runs), "execute": args.execute, "rounds": rounds,
                      "complete": sum((args.output / r.run_name() / "complete.json").exists() for r in runs),
                      "plan": [{"run_name": r.run_name(), "world": r.world, "client_seed": r.client_seed,
                                "noise_seed": r.noise_seed} for r in runs]}, indent=2))
    if args.execute:
        from results.cia_frontier.signature_suppression.runner import execute_shared

        execute_shared(runs, [], args.output, args.max_parallel_clients, stall_minutes=args.stall_minutes,
                       worker=_worker)


if __name__ == "__main__":
    main()
