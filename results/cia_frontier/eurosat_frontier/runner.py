"""Plan by default. Training requires --execute; never launches remote workers itself."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path

from experiments.reproduce.matrix import Combo
from results.contest_at_scale.auc_frontier.eurosat_remove import HYPERPARAMS


# kw_only: Combo now ends with a defaulted field (dirichlet_alpha), so required
# subclass fields must be keyword-only. All construction sites already use keywords.
@dataclass(frozen=True, kw_only=True)
class FrontierCombo(Combo):
    alpha: float
    canonical_clients: int
    out_target: int | None
    noise_ratio: float
    partition_seed: int | None = None  # None: ``seed`` controls data layout too
    # Default-off WP0 logging (not part of run_name; training is unchanged): server-side
    # leave-one-out residual norms, plus per-round shadow increments and gradient norms.
    diagnostics: bool = False

    @property
    def layout_seed(self) -> int:
        return self.seed if self.partition_seed is None else self.partition_seed

    def runner_args(self, **kwargs):
        profile = {"alpha": self.alpha, "clients": self.canonical_clients,
                   "out_target": self.out_target}
        if self.partition_seed is not None:
            profile["partition_seed"] = self.partition_seed
        args = (*super().runner_args(**kwargs), "--partition-profile",
                json.dumps(profile, sort_keys=True))
        return (*args, "--log-loo-residuals") if self.diagnostics else args


ADJACENCIES = ("both", "in", "out")


def build_combos(*, alpha, seeds, targets, clients, privacy, ratios, pilot=False,
                 adjacency="both", out_targets=None, partition_seed=None, diagnostics=False):
    """Build trajectories. ``targets`` is the fixed panel every shared IN evaluates.

    ``adjacency``/``out_targets`` only select which trajectories this process
    trains (for sharding across machines); they never change the IN panel.
    """
    if adjacency not in ADJACENCIES:
        raise ValueError("Unknown adjacency selection")
    if partition_seed is not None and (isinstance(partition_seed, bool)
                                       or not isinstance(partition_seed, int) or partition_seed < 0):
        raise ValueError("partition_seed must be a non-negative integer")
    if out_targets is not None:
        if adjacency == "in" or pilot:
            raise ValueError("--out-targets requires OUT trajectories (adjacency both/out)")
        if not out_targets or len(set(out_targets)) != len(out_targets):
            raise ValueError("Provide distinct OUT targets")
        if not set(out_targets) <= set(targets):
            raise ValueError("OUT targets must belong to the fixed --targets panel")
    if pilot and adjacency != "both":
        raise ValueError("Alpha pilot trains only IN; adjacency selection does not apply")
    if not math.isfinite(alpha) or alpha <= 0 or clients < 2:
        raise ValueError("Require finite alpha > 0 and clients >= 2")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct seeds")
    if not targets or len(set(targets)) != len(targets) or any(t < 0 or t >= clients for t in targets):
        raise ValueError("Targets must be distinct canonical client IDs")
    if privacy not in ("vanilla", "global-dp", "metric-privacy"):
        raise ValueError("Unknown privacy mode")
    if pilot and privacy != "vanilla":
        raise ValueError("Alpha pilot must be vanilla")
    if privacy == "vanilla":
        ratios = [0.0]
    elif not ratios or len(set(ratios)) != len(ratios) or any(not math.isfinite(r) or r <= 0 for r in ratios):
        raise ValueError("Provide distinct finite positive noise ratios")
    if pilot:
        selected = [None]
    else:
        outs = [t for t in targets if out_targets is None or t in out_targets]
        selected = ([None] if adjacency != "out" else []) + (outs if adjacency != "in" else [])
    layout = "" if partition_seed is None else f"-p{partition_seed}"
    return [FrontierCombo(
        name_prefix=(f"eurosat-dirichlet-a{alpha!r}{layout}-"
                     f"{'in' if target is None else f'out-{target}'}-r{ratio!r}"),
        num_clients=clients - (target is not None), partition="non-iid", privacy=privacy,
        aggregation="fedavg", seed=seed,
        noise_multiplier=ratio * (clients - (target is not None)), hyperparams=HYPERPARAMS,
        data_module="results.cia_frontier.eurosat_frontier.data:create_data_module",
        model_module="experiments.reproduce.eurosat_cnn:create_model",
        alpha=alpha, canonical_clients=clients, out_target=target, noise_ratio=ratio,
        partition_seed=partition_seed, diagnostics=diagnostics,
    ) for ratio in ratios for seed in seeds for target in selected]


def round_rows(round_number, target, before, after):
    """One diagnostics measurement row: the usual losses plus the round's increments.

    ``before``/``after`` are ``cia.eval_model_with_gradients`` outputs for the models
    before and after round ``round_number``'s update. Gradient norms are taken at the
    start model, the point where the round's update is applied.
    """
    row = dict(round=round_number, target=target, aggregate_loss=after["aggregate_loss"],
               clean_loss=after["clean_loss"], noisy_loss=after["noisy_loss"],
               shadow_size=after["shadow_size"])
    for view in ("clean", "noisy"):
        row[f"{view}_loss_before"] = before[f"{view}_loss"]
        row[f"{view}_increment"] = after[f"{view}_loss"] - before[f"{view}_loss"]
        row[f"{view}_grad_norm_start"] = before[f"{view}_grad_norm"]
    return row


EUROSAT_DATA_MODULE = "results.cia_frontier.eurosat_frontier.data:create_data_module"


def data_package(combo):
    """Python module behind ``combo.data_module``.

    It must provide ``create_data_module`` (training), ``base_module(alpha,
    partition_seed)`` (canonical module for shadow views) and ``partition_summary``.
    """
    import importlib
    return importlib.import_module(combo.data_module.split(":", 1)[0])


def release_norm(previous_state, state):
    """L2 norm of a released global-model update over all floating state entries."""
    import torch
    squared = sum(float(torch.sum((state[key].double() - previous_state[key].double()) ** 2))
                  for key in state if state[key].is_floating_point())
    count = sum(state[key].numel() for key in state if state[key].is_floating_point())
    return squared ** 0.5, count


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def execute(combos, targets, output, max_parallel_clients, pilot=False):
    """Lock each trajectory across local agents; independent runs can proceed in parallel."""
    import fcntl
    import hashlib
    import tempfile
    for combo in combos:
        identity = str((output / combo.run_name()).resolve()).encode()
        lock = Path(tempfile.gettempdir()) / ("auc-frontier-" + hashlib.sha256(identity).hexdigest() + ".lock")
        with lock.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            _execute([combo], targets, output, max_parallel_clients, pilot)


def _execute(combos, targets, output, max_parallel_clients, pilot=False):
    # Imports are delayed so planning cannot load data or initialize training.
    from experiments.cia.iter_combos import iter_combos
    from experiments.cia import cia
    import torch
    from experiments.cia.datasets.shadow import ShadowDataModule
    from metricdp_pytorch.utils.noisy_dataset import NoisyDataModule
    from metricdp_pytorch.utils.device import resolve_device

    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device()
    for combo in combos:
        folder = output / combo.run_name()
        folder.mkdir(parents=True, exist_ok=True)
        chosen = [] if pilot else (targets if combo.out_target is None else [combo.out_target])
        manifest = {"alpha": combo.alpha, "seed": combo.seed, "privacy": combo.privacy,
                    "noise_ratio": combo.noise_ratio,
                    "clients": combo.canonical_clients, "out_target": combo.out_target,
                    "targets": list(chosen), "rounds": combo.hyperparams.rounds,
                    "pilot": pilot, "schema": 1, "hyperparams": asdict(combo.hyperparams),
                    "run_name": combo.run_name(), "score_direction": "lower loss indicates IN"}
        if combo.partition_seed is not None:
            manifest["partition_seed"] = combo.partition_seed
        if combo.diagnostics:
            manifest["diagnostics"] = True
        if combo.data_module != EUROSAT_DATA_MODULE:
            manifest["data_module"] = combo.data_module
        manifest_path = folder / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"Manifest mismatch: {folder}; use a separate output directory")
        atomic_json(manifest_path, manifest)
        report = folder / "measurements.json"
        rounds = tuple(range(1, combo.hyperparams.rounds + 1))
        # Diagnostics also checkpoint the initial model (round 0) for round 1's increment.
        checkpoint_rounds = (0, *rounds) if combo.diagnostics else rounds
        expected = {(r, t) for r in rounds for t in chosen}
        rows = json.loads(report.read_text()) if report.exists() else []
        if (folder / "complete.json").exists() and {(r["round"], r["target"]) for r in rows} == expected:
            continue
        package = data_package(combo)
        atomic_json(folder / "partitions.json", package.partition_summary(
            combo.alpha, combo.canonical_clients, combo.layout_seed,
        ))
        import os
        import subprocess
        import sys
        revision = os.environ.get("METRICDP_SOURCE_COMMIT")
        if revision is None:
            revision = subprocess.run(
                ["git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        atomic_json(folder / "provenance.json", {
            "commit": revision, "python": sys.version, "device": str(device),
        })
        # A partial evaluation may have consumed checkpoints: retrain the entire
        # trajectory, rather than mixing measurements from separate executions.
        import time
        started = time.monotonic()
        for _, success, paths in iter_combos(
            [combo], output_dir=folder, max_parallel_clients=max_parallel_clients,
            force=True, log=print, checkpoint_rounds=checkpoint_rounds,
        ):
            if not success:
                raise RuntimeError(f"Training failed: {combo.run_name()}")
            trained = time.monotonic()
            print(f"[FRONTIER] training finished in {trained - started:.1f}s; "
                  f"evaluating {len(chosen)} target(s) x {len(rounds)} rounds", flush=True)
            # Evaluation data (shadow subsets, server test split) follows the
            # layout seed, so every training seed is scored on identical records.
            eval_combo = replace(combo, seed=combo.layout_seed)
            shadows = {}
            for target in chosen:
                base = package.base_module(combo.alpha, partition_seed=combo.partition_seed)
                kwargs = dict(num_clients=combo.canonical_clients, target_partition_id=target,
                              shadow_fraction=0.10, partition_mode="non-iid", partition_profile="auto")
                shadows[target] = (ShadowDataModule(base, **kwargs),
                                   ShadowDataModule(NoisyDataModule(base, std_fraction=0.20), **kwargs))
            rows = []
            previous = {}  # target -> evaluation of the previous checkpoint (diagnostics)
            releases, previous_state = [], None  # diagnostics: released-update norms
            for round_number, path in zip(checkpoint_rounds, paths, strict=True):
                if combo.diagnostics:
                    state = torch.load(path, map_location="cpu", weights_only=True)
                    if previous_state is not None:
                        norm, count = release_norm(previous_state, state)
                        releases.append(dict(round=round_number, update_norm=norm,
                                             parameter_count=count))
                    previous_state = state
                for target, (clean, noisy) in shadows.items():
                    views = dict(clean_data_module=clean, noisy_data_module=noisy,
                                 device=device, combo=eval_combo)
                    if combo.diagnostics:
                        after = cia.eval_model_with_gradients(path, **views)
                        if round_number > 0:
                            rows.append(round_rows(round_number, target, previous[target], after))
                        previous[target] = after
                    else:
                        aggregate, clean_loss, noisy_loss, size = cia.eval_model(path, **views)
                        rows.append(dict(round=round_number, target=target,
                                         aggregate_loss=aggregate, clean_loss=clean_loss,
                                         noisy_loss=noisy_loss, shadow_size=size))
                atomic_json(report, rows)
                path.unlink()  # only after ALL targets at this round are persisted
                print(f"[FRONTIER EVAL {round_number}/{len(rounds)}] targets={len(chosen)} "
                      f"elapsed={time.monotonic() - trained:.1f}s", flush=True)
            if combo.diagnostics:
                atomic_json(folder / "releases.json", releases)
            finished = time.monotonic()
            atomic_json(folder / "complete.json", {
                "complete": True, "training_seconds": round(trained - started, 1),
                "evaluation_seconds": round(finished - trained, 1),
            })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=list(range(20)),
                        help="Fixed target panel; every shared IN evaluates all of them")
    parser.add_argument("--adjacency", choices=ADJACENCIES, default="both",
                        help="Train only IN, only OUT, or both (sharding; panel unchanged)")
    parser.add_argument("--out-targets", type=int, nargs="+",
                        help="Subset of --targets whose OUT trajectories to train")
    parser.add_argument("--clients", type=int, default=48)
    parser.add_argument("--privacy", choices=("vanilla", "global-dp", "metric-privacy"), required=True)
    parser.add_argument("--ratios", type=float, nargs="+")
    parser.add_argument("--partition-seed", type=int,
                        help="Fix data layout independently of --seeds (training randomness only)")
    parser.add_argument("--alpha-pilot", action="store_true")
    parser.add_argument("--diagnostics", action="store_true",
                        help="log leave-one-out residual norms (server) and per-round shadow "
                             "increments and gradient norms (attack); training is unchanged")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("results/cia_frontier/eurosat_frontier/results"))
    args = parser.parse_args()
    if args.max_parallel_clients < 1:
        parser.error("--max-parallel-clients must be positive")
    combos = build_combos(alpha=args.alpha, seeds=args.seeds, targets=args.targets,
                          clients=args.clients, privacy=args.privacy, ratios=args.ratios,
                          pilot=args.alpha_pilot, adjacency=args.adjacency,
                          out_targets=args.out_targets, partition_seed=args.partition_seed,
                          diagnostics=args.diagnostics)
    print(json.dumps({"training_runs": len(combos), "execute": args.execute,
                      "target_panel": args.targets, "adjacency": args.adjacency,
                      "rounds_recorded": [1, HYPERPARAMS.rounds],
                      "run_names": [c.run_name() for c in combos]}, indent=2))
    if args.execute:
        execute(combos, args.targets, args.output, args.max_parallel_clients, args.alpha_pilot)


if __name__ == "__main__":
    main()
