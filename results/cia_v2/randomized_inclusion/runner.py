"""Randomized-inclusion CIA audit runs (CIA roadmap v2, Section 2 item 3). Plans unless --execute.

EuroSAT, label-Dirichlet alpha 0.3, the seed-42 layout, 48 canonical clients, 50 rounds, FedAvg
with equal weights. Thirty-two candidates (fixed below) are IN in half of the replicates; the other
16 clients always take part. In the fixed-count design (default) every replicate includes
exactly 16 candidates, so every run has 32 active clients and a public denominator. (Owner
delegation 2026-10-17, ESCALATION.md: 32 candidates instead of the roadmap's 16, for power.)

Randomness: the layout, the initial model, the server test split and the shadow samples are
public and shared (``layout-seed`` = ``init-seed`` = 42). Each replicate draws its own
``client_seed`` and ``noise_seed`` from ``(design_seed, replicate)``, so no two runs of one
configuration share private randomness; the same replicate of two configurations shares them
(allowed across configurations, to reduce the variance of differences).

Each run scores every candidate's shadow sample (10% of its training records, stratified,
without augmentation) at rounds 0, 5, ..., 50, plus optional label-matched public surrogates.

``--dataset alzheimer`` (owner decision 8, 2026-10-20) runs the same design on Alzheimer MRI with its own layout,
candidates, model and shadow fraction (``PROFILES``); EuroSAT is the default and its plans and manifests are unchanged.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np

from experiments.cia.inclusion import balanced_design
from experiments.reproduce.matrix import Combo
from experiments.reproduce.matrix import Hyperparams
from results.contest_at_scale.auc_frontier.alzheimer_remove import HYPERPARAMS as ALZHEIMER_HYPERPARAMS
from results.contest_at_scale.auc_frontier.eurosat_remove import HYPERPARAMS

ALPHA = 0.3
SLOTS = 48
LAYOUT_SEED = 42
ROUNDS = 50
CHECKPOINT_ROUNDS = tuple(range(0, ROUNDS + 1, 5))
SHADOW_FRACTION = 0.10
FRONTIER_RATIO = 0.001546  # noise std / clipping norm at the v1 EuroSAT frontier (WP2, WP6)
DATA_MODULE = "results.cia_v2.randomized_inclusion.data:create_data_module"
MODEL_MODULE = "experiments.reproduce.eurosat_cnn:create_model"
PRIVACY_MODES = ("vanilla", "global-dp", "metric-privacy", "private-median")
# WP3's private median (A1): initial median client distance and its public bounds. v1's EuroSAT metric-privacy runs at
# 48 clients logged pairwise client distances with median 0.39-0.53, so client-to-global distances near 0.3.
MEDIAN_QUANTILE, MEDIAN_QUANTILE_MIN, MEDIAN_QUANTILE_MAX = 0.3, 0.03, 1.0
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = 1


def select_candidates(sizes, others=27):
    """The top decile by record count (5 of 48), then ``others`` of the other 43 by size quantile.

    The others are sorted by ascending size and picked at ranks ``round((k + 0.5) * 43 / others)``,
    k = 0..others-1 (Python rounding, half to even); ties break by client ID.
    """
    by_size = sorted(sizes, key=lambda t: (-sizes[t], t))
    top = tuple(by_size[:5])
    rest = sorted(by_size[5:], key=lambda t: (sizes[t], t))
    return top, tuple(rest[round((k + 0.5) * len(rest) / others)] for k in range(others))


# Fixed before any run from the seed-42 layout (partition_summary(0.3, 48, 42)): top decile 23 (1,150 records),
# 7 (1,089), 40 (917), 18 (873), 33 (864); then 27 by size quantile, 100 to 809 records. The 16-candidate panel of
# the roadmap's specification (top decile plus 1, 22, 16, 31, 47, 28, 27, 8, 29, 25, 43) is a subset.
TOP_DECILE = (23, 7, 40, 18, 33)
CANDIDATES = TOP_DECILE + (17, 1, 44, 22, 46, 39, 16, 6, 31, 20, 5, 47, 14, 28, 36, 27, 24, 2, 8, 34, 29, 35, 9, 25,
                           13, 43, 37)


# Alzheimer MRI (owner decision 8, setup chosen 2026-10-21 after results/cia_v2/alzheimer/ESCALATION.md: alpha 1.0,
# 100 rounds). Fixed before any gated run from the seed-42 layout with at least 10 records per client
# (partition_summary(1.0, 48, 42, min_records=10)) by the same rule: top decile 4 (311 records), 29 (255), 31 (236),
# 22 (223), 15 (208); then 27 by size quantile, 19 to 203 records.
ALZHEIMER_TOP_DECILE = (4, 29, 31, 22, 15)
ALZHEIMER_CANDIDATES = ALZHEIMER_TOP_DECILE + (11, 21, 2, 38, 12, 37, 10, 13, 6, 35, 42, 47, 24, 9, 20, 27, 5, 39, 23, 45,
                                               32, 18, 43, 41, 30, 26, 0)


@dataclass(frozen=True)
class DatasetProfile:
    """What differs between datasets; the design (slots, alpha, layout seed, rounds, R) does not."""

    name: str
    data_module: str
    model_module: str
    hyperparams: Hyperparams
    top_decile: tuple[int, ...]
    candidates: tuple[int, ...]
    shadow_fraction: float
    median_quantile: tuple[float, float, float]  # private median: initial, minimum, maximum
    alpha: float = ALPHA
    rounds: int = ROUNDS
    checkpoint_every: int = 5  # Alzheimer: every 10 of 100 rounds, so the attacker sees 11 checkpoints as on EuroSAT


PROFILES = {
    "eurosat": DatasetProfile("eurosat", DATA_MODULE, MODEL_MODULE, HYPERPARAMS, TOP_DECILE, CANDIDATES,
                              SHADOW_FRACTION, (MEDIAN_QUANTILE, MEDIAN_QUANTILE_MIN, MEDIAN_QUANTILE_MAX)),
    # Shadow fraction 25%: at 10% the smallest candidates get 1 image (median 8); at 25%, 3 (median 20). The private-
    # median quantile is set from the accuracy calibration's metric-privacy distances (results/cia_v2/alzheimer/).
    "alzheimer": DatasetProfile("alzheimer", DATA_MODULE, "experiments.reproduce.paper_cnn:create_model",
                                ALZHEIMER_HYPERPARAMS, ALZHEIMER_TOP_DECILE, ALZHEIMER_CANDIDATES, 0.25,
                                (MEDIAN_QUANTILE, MEDIAN_QUANTILE_MIN, MEDIAN_QUANTILE_MAX),
                                alpha=1.0, rounds=100, checkpoint_every=10),
}


def design_matrix(replicates, *, design_seed, fixed_count=True):
    """``(len(CANDIDATES), replicates)`` inclusion matrix, rows in ``CANDIDATES`` order."""
    return balanced_design(len(CANDIDATES), replicates, seed=design_seed, fixed_count=fixed_count)


def replicate_seeds(design_seed, replicate):
    """``(client_seed, noise_seed)`` of one replicate: independent draws, shared across configurations."""
    client_seed, noise_seed = np.random.SeedSequence([int(design_seed), int(replicate)]).generate_state(2)
    return int(client_seed), int(noise_seed)


@dataclass(frozen=True, kw_only=True)
class InclusionRun(Combo):
    config: str
    replicate: int
    replicates: int
    design_seed: int
    included: tuple[int, ...]
    active: tuple[int, ...]
    client_seed: int
    noise_seed: int
    noise_ratio: float
    dataset: str = "eurosat"
    alpha: float = ALPHA  # the profile's (EuroSAT 0.3, Alzheimer 1.0); other values for the AZ diagnosis only

    @property
    def profile(self):
        return PROFILES[self.dataset]

    def runner_args(self, **kwargs):
        profile = {"alpha": self.alpha, "clients": SLOTS, "partition_seed": LAYOUT_SEED, "active": list(self.active)}
        if self.dataset != "eurosat":
            profile["dataset"] = self.dataset
        args = (*super().runner_args(**kwargs),
                "--partition-profile", json.dumps(profile, sort_keys=True),
                "--layout-seed", str(LAYOUT_SEED), "--init-seed", str(LAYOUT_SEED),
                "--client-seed", str(self.client_seed), "--noise-seed", str(self.noise_seed))
        if self.privacy == "private-median":
            # The fixed active count is public, so it is the mechanism's slot denominator.
            initial, minimum, maximum = self.profile.median_quantile
            args = (*args, "--num-slots", str(self.num_clients), "--median-quantile", str(initial),
                    "--median-quantile-min", str(minimum), "--median-quantile-max", str(maximum))
        return (*args, "--log-noise-diagnostics") if self.privacy != "vanilla" else args


def build_runs(*, config, privacy, noise_ratio, replicates, design_seed, replicate_subset=None, rounds=None,
               dataset="eurosat", alpha=None):
    if privacy not in PRIVACY_MODES:
        raise ValueError(f"privacy must be one of {PRIVACY_MODES}")
    if privacy == "vanilla" and noise_ratio:
        raise ValueError("vanilla takes no noise")
    if privacy != "vanilla" and not noise_ratio > 0:
        raise ValueError("DP runs need a positive noise ratio")
    profile = PROFILES[dataset]
    rounds = profile.rounds if rounds is None else rounds
    alpha = profile.alpha if alpha is None else alpha
    design = design_matrix(replicates, design_seed=design_seed)
    selected = range(replicates) if replicate_subset is None else sorted(set(replicate_subset))
    always_in = sorted(set(range(SLOTS)) - set(profile.candidates))
    runs = []
    for j in selected:
        if not 0 <= j < replicates:
            raise ValueError(f"replicate {j} outside the design")
        included = tuple(sorted(c for c, row in zip(profile.candidates, design) if row[j]))
        active = tuple(sorted(always_in + list(included)))
        client_seed, noise_seed = replicate_seeds(design_seed, j)
        runs.append(InclusionRun(
            name_prefix=f"ri-{config}-d{design_seed}-rep{j:02d}", num_clients=len(active), partition="non-iid",
            privacy=privacy, aggregation="fedavg", seed=LAYOUT_SEED,
            # Noise std r C, independent of the active count (compute_stdv divides by it).
            noise_multiplier=noise_ratio * len(active), hyperparams=replace(profile.hyperparams, rounds=rounds),
            data_module=profile.data_module, model_module=profile.model_module, aggregation_weighting="equal",
            config=config, replicate=j, replicates=replicates, design_seed=design_seed, included=included, active=active,
            client_seed=client_seed, noise_seed=noise_seed, noise_ratio=noise_ratio, dataset=dataset, alpha=alpha,
        ))
    return runs


def without_augmentation(dataset):
    """The same records with any training augmentation switched off (through Subset wrappers)."""
    from torch.utils.data import Subset

    if isinstance(dataset, Subset):
        return Subset(without_augmentation(dataset.dataset), dataset.indices)
    transform = getattr(dataset, "transform", None)
    if dataclasses.is_dataclass(transform) and getattr(transform, "augment", False):
        clone = copy.copy(dataset)
        clone.transform = dataclasses.replace(transform, augment=False)
        return clone
    return dataset


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def checkpoint_rounds(rounds, every=5):
    """Rounds 0, ``every``, 2 ``every``, ... up to ``rounds``, plus the last round."""
    return sorted({r for r in range(0, rounds + 1, every)} | {rounds})


def manifest(run, surrogates):
    pinned = _manifest(run, surrogates)
    if run.privacy == "private-median":
        initial, minimum, maximum = run.profile.median_quantile
        pinned["median_quantile"] = {"initial": initial, "minimum": minimum, "maximum": maximum,
                                     "num_slots": run.num_clients}
    if run.dataset != "eurosat":
        pinned["dataset"] = run.dataset
    if run.alpha != ALPHA:
        pinned["alpha"] = run.alpha
    return pinned


def _manifest(run, surrogates):
    return {"schema": SCHEMA, "run_name": run.run_name(), "config": run.config, "privacy": run.privacy,
            "noise_ratio": run.noise_ratio, "noise_multiplier": run.noise_multiplier, "replicate": run.replicate,
            "replicates": run.replicates, "design_seed": run.design_seed, "candidates": list(run.profile.candidates),
            "top_decile": list(run.profile.top_decile),
            "included": list(run.included), "active": list(run.active), "slots": SLOTS,
            "layout_seed": LAYOUT_SEED, "init_seed": LAYOUT_SEED, "client_seed": run.client_seed,
            "noise_seed": run.noise_seed,
            "checkpoint_rounds": checkpoint_rounds(run.hyperparams.rounds, run.profile.checkpoint_every),
            "shadow_fraction": run.profile.shadow_fraction, "shadow_augmentation": False, "surrogates": surrogates,
            "hyperparams": asdict(run.hyperparams), "score_direction": "lower loss indicates IN"}


def _loss_and_accuracy(model, loader, device):
    import torch

    from experiments.reproduce.paper_loss import sparse_categorical_cross_entropy
    from metricdp_pytorch.utils.device import to_device

    model.to(device)
    model.eval()
    total, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = to_device(images, device), to_device(labels, device)
            probabilities = model(images)
            total += sparse_categorical_cross_entropy(probabilities, labels, reduction="sum").item()
            correct += int((probabilities.argmax(dim=1) == labels).sum())
            count += len(labels)
    return total / count, correct / count


def evaluation_sets(batch_size, surrogates, profile=PROFILES["eurosat"]):
    """Public test loader, and per candidate its shadow loader and optional surrogate loader."""
    from torch.utils.data import DataLoader, Subset

    from experiments.cia.datasets.shadow import ShadowDataModule, _labels_from_dataset
    from results.cia_v2.randomized_inclusion.surrogate import label_matched_indices

    base = _base_module(profile.name)
    _, test_loader = base.server_loaders(batch_size=batch_size, seed=LAYOUT_SEED)
    test_labels = np.asarray(_labels_from_dataset(test_loader.dataset))
    sets = {}
    for candidate in profile.candidates:
        shadow = ShadowDataModule(base, num_clients=SLOTS, target_partition_id=candidate,
                                  shadow_fraction=profile.shadow_fraction, partition_mode="non-iid",
                                  partition_profile="auto")
        records = without_augmentation(shadow.target_shadow_loader(batch_size=batch_size, seed=LAYOUT_SEED).dataset)
        loaders = {"shadow": DataLoader(records, batch_size=batch_size, shuffle=False)}
        if surrogates:
            counts = np.bincount(_labels_from_dataset(records))
            chosen = label_matched_indices(test_labels, target_counts=dict(enumerate(counts.tolist())),
                                           seed=LAYOUT_SEED + candidate)
            loaders["surrogate"] = DataLoader(Subset(test_loader.dataset, chosen.tolist()),
                                              batch_size=batch_size, shuffle=False)
        sets[candidate] = loaders
    return test_loader, sets


def _base_module(dataset="eurosat"):
    from results.cia_v2.randomized_inclusion.data import base_module

    return base_module(ALPHA, partition_seed=LAYOUT_SEED, dataset=dataset)


def execute_run(run, output, max_parallel_clients, surrogates=False):
    """Train one replicate, score every candidate at every checkpoint, then delete the checkpoints."""
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
    pinned = manifest(run, surrogates)
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
    # A partial evaluation may have consumed checkpoints: always retrain the whole run.
    for _, success, paths in iter_combos([run], output_dir=folder, max_parallel_clients=max_parallel_clients,
                                         force=True, log=print, checkpoint_rounds=rounds):
        if not success:
            raise RuntimeError(f"Training failed: {run.run_name()}")
        trained = time.monotonic()
        test_loader, sets = evaluation_sets(run.hyperparams.batch_size, surrogates, run.profile)
        rows = []
        for round_number, checkpoint in zip(rounds, paths, strict=True):
            model = load_model(run.model_module)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            test_loss, test_accuracy = _loss_and_accuracy(model, test_loader, device)
            for candidate, loaders in sets.items():
                row = {"round": round_number, "candidate": candidate, "included": candidate in run.included,
                       "test_loss": test_loss, "test_accuracy": test_accuracy}
                for view, loader in loaders.items():
                    row[f"{view}_loss"], row[f"{view}_accuracy"] = _loss_and_accuracy(model, loader, device)
                rows.append(row)
            atomic_json(folder / "measurements.json", rows)
            checkpoint.unlink()  # only after every candidate at this round is persisted
        atomic_json(folder / "complete.json", {"complete": True, "training_seconds": round(trained - started, 1),
                                               "evaluation_seconds": round(time.monotonic() - trained, 1)})


def _worker(run, targets, output, max_parallel_clients, surrogates=False):
    """Child-process entry point for the shared stall-guarded executor (``targets`` unused)."""
    del targets
    execute_run(run, output, max_parallel_clients, surrogates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="configuration name (part of run names)")
    parser.add_argument("--privacy", choices=PRIVACY_MODES, required=True)
    parser.add_argument("--dataset", choices=tuple(PROFILES), default="eurosat")
    parser.add_argument("--alpha", type=float, help="label-Dirichlet alpha (default: the dataset profile's)")
    parser.add_argument("--rounds", type=int, help="training rounds (default: the dataset profile's)")
    parser.add_argument("--noise-ratio", type=float, default=0.0, help="noise std / clipping norm (DP only)")
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--design-seed", type=int, required=True)
    parser.add_argument("--replicate-subset", type=int, nargs="+", help="train only these replicates (sharding)")
    parser.add_argument("--surrogates", action="store_true", help="also score label-matched public surrogates")
    parser.add_argument("--smoke-rounds", type=int, help="1-5 rounds into an --output outside results/")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=8)
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
    runs = build_runs(config=args.config, privacy=args.privacy, noise_ratio=args.noise_ratio,
                      replicates=args.replicates, design_seed=args.design_seed,
                      replicate_subset=args.replicate_subset, rounds=rounds, dataset=args.dataset, alpha=args.alpha)
    print(json.dumps({"runs": len(runs), "execute": args.execute, "rounds": runs[0].hyperparams.rounds if runs else rounds,
                      "complete": sum((args.output / r.run_name() / "complete.json").exists() for r in runs),
                      "plan": [{"run_name": r.run_name(), "included": r.included, "client_seed": r.client_seed,
                                "noise_seed": r.noise_seed, "noise_multiplier": r.noise_multiplier} for r in runs]},
                     indent=2))
    if args.execute:
        from functools import partial

        from results.cia_frontier.signature_suppression.runner import execute_shared

        execute_shared(runs, [], args.output, args.max_parallel_clients, stall_minutes=args.stall_minutes,
                       worker=partial(_worker, surrogates=args.surrogates))


if __name__ == "__main__":
    main()
