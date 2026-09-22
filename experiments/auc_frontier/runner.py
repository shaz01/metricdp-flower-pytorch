"""Plan by default. Training requires --execute; never launches remote workers itself."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from experiments.reproduce.matrix import Combo
from experiments.cia.scripts.eurosat_remove import HYPERPARAMS


@dataclass(frozen=True)
class FrontierCombo(Combo):
    alpha: float
    canonical_clients: int
    out_target: int | None
    noise_ratio: float

    def runner_args(self, **kwargs):
        return (*super().runner_args(**kwargs), "--partition-profile", json.dumps({
            "alpha": self.alpha, "clients": self.canonical_clients,
            "out_target": self.out_target,
        }, sort_keys=True))


def build_combos(*, alpha, seeds, targets, clients, privacy, ratios, pilot=False):
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
    return [FrontierCombo(
        name_prefix=f"eurosat-dirichlet-a{alpha!r}-{'in' if target is None else f'out-{target}'}-r{ratio!r}",
        num_clients=clients - (target is not None), partition="non-iid", privacy=privacy,
        aggregation="fedavg", seed=seed,
        noise_multiplier=ratio * (clients - (target is not None)), hyperparams=HYPERPARAMS,
        data_module="experiments.auc_frontier.data:create_data_module",
        model_module="experiments.reproduce.eurosat_cnn:create_model",
        alpha=alpha, canonical_clients=clients, out_target=target, noise_ratio=ratio,
    ) for ratio in ratios for seed in seeds for target in ([None] if pilot else [None, *targets])]


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
    from experiments.cia.datasets.shadow import ShadowDataModule
    from metricdp_pytorch.utils.noisy_dataset import NoisyDataModule
    from metricdp_pytorch.utils.device import resolve_device
    from experiments.auc_frontier.data import DirichletEuroSAT, partition_summary

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
        manifest_path = folder / "manifest.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"Manifest mismatch: {folder}; use a separate output directory")
        atomic_json(manifest_path, manifest)
        report = folder / "measurements.json"
        rounds = tuple(range(1, combo.hyperparams.rounds + 1))
        expected = {(r, t) for r in rounds for t in chosen}
        rows = json.loads(report.read_text()) if report.exists() else []
        if (folder / "complete.json").exists() and {(r["round"], r["target"]) for r in rows} == expected:
            continue
        atomic_json(folder / "partitions.json", partition_summary(
            combo.alpha, combo.canonical_clients, combo.seed,
        ))
        import subprocess
        import sys
        revision = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        atomic_json(folder / "provenance.json", {
            "commit": revision, "python": sys.version, "device": str(device),
        })
        # A partial evaluation may have consumed checkpoints: retrain the entire
        # trajectory, rather than mixing measurements from separate executions.
        for _, success, paths in iter_combos(
            [combo], output_dir=folder, max_parallel_clients=max_parallel_clients,
            force=True, log=print, checkpoint_rounds=rounds,
        ):
            if not success:
                raise RuntimeError(f"Training failed: {combo.run_name()}")
            shadows = {}
            for target in chosen:
                base = DirichletEuroSAT(combo.alpha)
                kwargs = dict(num_clients=combo.canonical_clients, target_partition_id=target,
                              shadow_fraction=0.10, partition_mode="non-iid", partition_profile="auto")
                shadows[target] = (ShadowDataModule(base, **kwargs),
                                   ShadowDataModule(NoisyDataModule(base, std_fraction=0.20), **kwargs))
            rows = []
            for round_number, path in zip(rounds, paths, strict=True):
                for target, (clean, noisy) in shadows.items():
                    aggregate, clean_loss, noisy_loss, size = cia.eval_model(
                        path, clean_data_module=clean, noisy_data_module=noisy,
                        device=device, combo=combo,
                    )
                    rows.append(dict(round=round_number, target=target,
                                     aggregate_loss=aggregate, clean_loss=clean_loss,
                                     noisy_loss=noisy_loss, shadow_size=size))
                atomic_json(report, rows)
                path.unlink()  # only after ALL targets at this round are persisted
            atomic_json(folder / "complete.json", {"complete": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--targets", type=int, nargs="+", default=list(range(20)))
    parser.add_argument("--clients", type=int, default=48)
    parser.add_argument("--privacy", choices=("vanilla", "global-dp", "metric-privacy"), required=True)
    parser.add_argument("--ratios", type=float, nargs="+")
    parser.add_argument("--alpha-pilot", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-parallel-clients", type=int, default=6)
    parser.add_argument("--output", type=Path, default=Path("results/new_auc_frontier_eurosat"))
    args = parser.parse_args()
    if args.max_parallel_clients < 1:
        parser.error("--max-parallel-clients must be positive")
    combos = build_combos(alpha=args.alpha, seeds=args.seeds, targets=args.targets,
                          clients=args.clients, privacy=args.privacy, ratios=args.ratios,
                          pilot=args.alpha_pilot)
    print(json.dumps({"training_runs": len(combos), "execute": args.execute,
                      "rounds_recorded": [1, HYPERPARAMS.rounds],
                      "run_names": [c.run_name() for c in combos]}, indent=2))
    if args.execute:
        execute(combos, args.targets, args.output, args.max_parallel_clients, args.alpha_pilot)


if __name__ == "__main__":
    main()
