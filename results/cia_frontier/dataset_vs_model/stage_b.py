"""Stage B/C of dataset_vs_model: accuracy, IN/OUT shadow losses and per-client influence.

One model (cifar10_cnn). Cells: {cifar10s, eurosat32} x {homogeneous, dirichlet a=0.3} x
{vanilla, global-dp x 4 ratios, metric-privacy x 4 ratios} x seeds. Both datasets have a
21,600-image pool (450 images per client at 48 clients).

Every Stage B run is an IN trajectory (all 48 canonical clients). After each round its
checkpoint is scored on the clean and noisy shadow sets of targets 0-9, using the
eurosat_frontier measurement path (experiments/cia/trajectories.py). A target's
clean shadow set is a deterministic stratified 10% subset of that target's own training
split, so the IN model trains on the shadow records. OUT trajectories (``--out-targets``)
drop one target and keep every other client's records unchanged; they score the same
target's identical shadow records (checked through shadows.json). Training also logs
per-client influence every round (``influence-*`` train metrics).

Plans by default; training requires --execute. Never launches remote workers itself.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path

from experiments.reproduce.matrix import Combo
from results.cia_frontier.dataset_vs_model.data import STAGE_B_DATASETS, STAGE_B_VIEW, stage_b_profile
from results.cia_frontier.dataset_vs_model.runner import HYPERPARAMS as _STAGE_A_HYPERPARAMS, MODELS

ROUNDS = 20
CLIENTS = 48
MODEL = "cifar10_cnn"
DIRICHLET_ALPHA = 0.3
RATIOS = (0.001, 0.0025, 0.004, 0.00625)  # contest_at_scale/cifar10's three, plus one below
PRIVACY = ("vanilla", "global-dp", "metric-privacy")
PARTITIONS = ("homogeneous", "dirichlet")
SEEDS = (42,)
TARGETS = tuple(range(10))
SHADOW_FRACTION = 0.10
NOISY_STD_FRACTION = 0.20
HYPERPARAMS = replace(_STAGE_A_HYPERPARAMS, rounds=ROUNDS)
DEFAULT_OUTPUT = Path("results/cia_frontier/dataset_vs_model/results/stage_b")

STAGES = ("b", "c")
CELLS = tuple((d, p) for d in STAGE_B_DATASETS for p in PARTITIONS)


@dataclass(frozen=True, kw_only=True)
class StageBCombo(Combo):
    dataset: str
    canonical_clients: int
    out_target: int | None
    noise_ratio: float
    train_subsample: int | None = None  # local smoke runs only

    @property
    def profile(self) -> str:
        return stage_b_profile(self.dataset, self.canonical_clients, self.out_target,
                               self.train_subsample)

    def runner_args(self, **kwargs):
        return (*super().runner_args(**kwargs), "--partition-profile", self.profile)


def cell_name(dataset: str, partition: str) -> str:
    return f"{dataset}+{partition}"


def arm_name(privacy: str, ratio: float | None) -> str:
    return "vanilla" if privacy == "vanilla" else f"{privacy}-r{ratio!r}"


def build_combos(*, seeds=SEEDS, cells=None, ratios=RATIOS, privacy=PRIVACY,
                 rounds=ROUNDS, clients=CLIENTS, alpha=DIRICHLET_ALPHA,
                 targets=TARGETS, out_targets=None, train_subsample=None, stage="b"):
    """IN trajectories by default; with ``out_targets``, only those OUT trajectories.

    ``stage`` only labels run names (``stage-<stage>-...``); Stage C reuses these builders.

    Noise multiplier = ratio x active clients (48 IN, 47 OUT), eurosat_frontier's convention.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seeds")
    if not ratios or any(not math.isfinite(r) or r <= 0 for r in ratios) or len(set(ratios)) != len(ratios):
        raise ValueError("Ratios must be distinct, finite and positive")
    if clients < 2 or rounds < 1 or alpha <= 0:
        raise ValueError("Require clients >= 2, rounds >= 1, alpha > 0")
    if not targets or len(set(targets)) != len(targets) or any(not 0 <= t < clients for t in targets):
        raise ValueError("Targets must be distinct canonical client IDs")
    if out_targets is not None and (not out_targets or len(set(out_targets)) != len(out_targets)
                                    or not set(out_targets) <= set(targets)):
        raise ValueError("OUT targets must be distinct members of the target panel")
    if stage not in STAGES:
        raise ValueError(f"Unknown stage label: {stage}")
    unknown = set(privacy) - set(PRIVACY)
    if unknown:
        raise ValueError(f"Unknown privacy modes: {sorted(unknown)}")
    wanted = [cell_name(*c) for c in CELLS] if cells is None else list(cells)
    bad = set(wanted) - {cell_name(*c) for c in CELLS}
    if bad or len(set(wanted)) != len(wanted):
        raise ValueError(f"Unknown or repeated cells: {sorted(bad) or wanted}")
    arms = [(p, None) if p == "vanilla" else (p, r) for p in privacy for r in ([None] if p == "vanilla" else ratios)]
    adjacencies = [None] if out_targets is None else list(out_targets)
    return [StageBCombo(
        name_prefix=(f"stage-{stage}-{dataset}-{partition}-{arm_name(priv, ratio)}-"
                     f"{'in' if target is None else f'out-{target}'}"),
        num_clients=clients - (target is not None), partition=partition, dirichlet_alpha=alpha,
        privacy=priv, aggregation="fedavg", seed=seed,
        noise_multiplier=0.0 if ratio is None else ratio * (clients - (target is not None)),
        hyperparams=replace(HYPERPARAMS, rounds=rounds),
        data_module=STAGE_B_VIEW, model_module=MODELS[MODEL], data_tag=dataset,
        log_client_influence=True,
        dataset=dataset, canonical_clients=clients, out_target=target,
        noise_ratio=0.0 if ratio is None else ratio, train_subsample=train_subsample,
    ) for dataset, partition in CELLS if cell_name(dataset, partition) in wanted
      for priv, ratio in arms for seed in seeds for target in adjacencies]


def manifest(combo: StageBCombo, targets) -> dict:
    chosen = list(targets) if combo.out_target is None else [combo.out_target]
    value = {"dataset": combo.dataset, "partition": combo.partition,
             "alpha": combo.dirichlet_alpha if combo.partition == "dirichlet" else None,
             "seed": combo.seed, "privacy": combo.privacy, "noise_ratio": combo.noise_ratio,
             "clients": combo.canonical_clients, "out_target": combo.out_target,
             "targets": chosen, "rounds": combo.hyperparams.rounds, "pilot": False, "schema": 1,
             "hyperparams": asdict(combo.hyperparams), "run_name": combo.run_name(),
             "shadow_fraction": SHADOW_FRACTION, "noisy_std_fraction": NOISY_STD_FRACTION,
             "score_direction": "lower loss indicates IN"}
    if combo.train_subsample is not None:
        value["train_subsample"] = combo.train_subsample
    return value


def _base(combo: StageBCombo):
    from results.cia_frontier.dataset_vs_model.data import AutoProfile, stage_b_base
    return AutoProfile(stage_b_base(json.loads(combo.profile), {}))


def shadow_modules(combo: StageBCombo, target: int, base=None):
    """Clean/noisy shadow modules built from the canonical (all-client) partitions.

    Independent of IN vs OUT by construction: only dataset, partition, alpha, seed and
    target enter, never the active client list.
    """
    from experiments.cia.datasets.shadow import ShadowDataModule
    from metricdp_pytorch.utils.noisy_dataset import NoisyDataModule
    if combo.max_client_samples:
        raise ValueError("Shadow sets assume uncapped clients (max_client_samples=0)")
    base = _base(combo) if base is None else base
    kwargs = dict(num_clients=combo.canonical_clients, target_partition_id=target,
                  shadow_fraction=SHADOW_FRACTION, partition_mode=combo.partition,
                  partition_profile="auto", dirichlet_alpha=combo.dirichlet_alpha)
    return (ShadowDataModule(base, **kwargs),
            ShadowDataModule(NoisyDataModule(base, std_fraction=NOISY_STD_FRACTION), **kwargs))


def root_indices(dataset) -> list[int]:
    """Indices into the underlying record pool, through nested Subsets/noise wrappers."""
    from torch.utils.data import Subset
    if isinstance(dataset, Subset):
        inner = root_indices(dataset.dataset)
        return [inner[i] for i in dataset.indices]
    from metricdp_pytorch.utils.noisy_dataset import NoisyDataset
    if isinstance(dataset, NoisyDataset):
        return root_indices(dataset.dataset)
    return list(range(len(dataset)))


def shadow_fingerprints(combo: StageBCombo, shadows: dict) -> dict:
    """Hash each target's shadow record IDs; check what the trajectory trains on."""
    from results.cia_frontier.dataset_vs_model.data import create_stage_b_view
    batch, seed = combo.hyperparams.batch_size, combo.seed
    view = create_stage_b_view({"partition-profile": combo.profile})
    output = {}
    for target, (clean, noisy) in shadows.items():
        clean_ids = root_indices(clean.target_shadow_loader(batch_size=batch, seed=seed).dataset)
        noisy_ids = root_indices(noisy.target_shadow_loader(batch_size=batch, seed=seed).dataset)
        if clean_ids != noisy_ids:
            raise ValueError(f"Clean and noisy shadow records differ for target {target}")
        if target in view.active_partition_ids:
            active = view.active_partition_ids.index(target)
            train, _ = view.client_loaders(active, num_partitions=view.num_active_partitions,
                                           partition_mode=combo.partition, batch_size=batch, seed=seed,
                                           dirichlet_alpha=combo.dirichlet_alpha)
            if not set(clean_ids) <= set(root_indices(train.dataset)):
                raise ValueError(f"IN trajectory does not train on target {target}'s shadow records")
        elif combo.out_target != target:
            raise ValueError(f"Target {target} missing from a trajectory that should include it")
        output[str(target)] = {"size": len(clean_ids), "in_training": target in view.active_partition_ids,
                               "sha256": hashlib.sha256(json.dumps(clean_ids).encode()).hexdigest()}
    return output


def execute(combos, targets, output: Path, max_parallel_clients: int):
    from experiments.cia.trajectories import execute_locked, execute_trajectory

    def run(combo):
        chosen = list(targets) if combo.out_target is None else [combo.out_target]
        base = _base(combo)
        execute_trajectory(
            combo, chosen=chosen, manifest=manifest(combo, targets), output=output,
            max_parallel_clients=max_parallel_clients,
            partition_summary=lambda: partition_summary(combo, base),
            make_shadows=lambda target: shadow_modules(combo, target, base),
            eval_seed=combo.seed,  # one seed controls layout and training, as in eurosat_frontier
            shadow_fingerprints=lambda shadows: shadow_fingerprints(combo, shadows),
        )
    execute_locked(combos, run, output)


def partition_summary(combo: StageBCombo, base) -> dict:
    clients = []
    for client in range(combo.canonical_clients):
        train, test = base.client_loaders(client, num_partitions=combo.canonical_clients,
                                          partition_mode=combo.partition, batch_size=32,
                                          seed=combo.seed, dirichlet_alpha=combo.dirichlet_alpha)
        clients.append({"target": client, "train_records": len(train.dataset),
                        "test_records": len(test.dataset)})
    return {"dataset": combo.dataset, "partition": combo.partition, "seed": combo.seed,
            "alpha": combo.dirichlet_alpha if combo.partition == "dirichlet" else None,
            "clients": clients}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--cells", nargs="+", choices=[cell_name(*c) for c in CELLS],
                        help="Subset of cells to train (sharding across machines)")
    parser.add_argument("--ratios", type=float, nargs="+", default=list(RATIOS))
    parser.add_argument("--privacy", nargs="+", choices=PRIVACY, default=list(PRIVACY))
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--targets", type=int, nargs="+", default=list(TARGETS),
                        help="Fixed target panel every IN trajectory measures")
    parser.add_argument("--out-targets", type=int, nargs="+",
                        help="Train only these OUT trajectories (each must be in --targets)")
    parser.add_argument("--with-in", action="store_true",
                        help="With --out-targets, also train the IN trajectory (shard IN and OUT together)")
    parser.add_argument("--stage", choices=STAGES, default="b", help="Run-name label only")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=6)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_parallel_clients < 1:
        parser.error("--max-parallel-clients must be positive")
    if args.with_in and not args.out_targets:
        parser.error("--with-in requires --out-targets")
    kwargs = dict(seeds=args.seeds, cells=args.cells, ratios=args.ratios, privacy=args.privacy,
                  rounds=args.rounds, targets=args.targets, stage=args.stage)
    combos = ((build_combos(**kwargs) if args.with_in else [])
              + build_combos(**kwargs, out_targets=args.out_targets))
    print(json.dumps({"training_runs": len(combos), "execute": args.execute,
                      "target_panel": args.targets, "out_targets": args.out_targets,
                      "run_names": [c.run_name() for c in combos]}, indent=2), flush=True)
    if args.execute:
        execute(combos, args.targets, args.output.resolve(), args.max_parallel_clients)


if __name__ == "__main__":
    main()
