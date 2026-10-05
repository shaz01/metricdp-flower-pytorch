"""Shared IN/OUT trajectory execution for the cia_frontier experiments.

Generalized from eurosat_frontier's runner (unchanged behaviour there): train one
trajectory with a checkpoint every round, score every chosen target's clean and
noisy shadow sets on each checkpoint, write ``measurements.json`` atomically and
delete each checkpoint only after all targets at that round are saved.

Layout of one trajectory folder: ``manifest.json`` (immutable plan),
``partitions.json``, ``provenance.json``, the training run JSON,
``measurements.json`` (rows: round, target, aggregate_loss, clean_loss,
noisy_loss, shadow_size), optional ``shadows.json`` (per-target shadow
fingerprints) and ``complete.json``.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import json
from pathlib import Path


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def execute_locked(combos, run, output):
    """Lock each trajectory across local agents; independent runs proceed in parallel."""
    import fcntl
    import hashlib
    import tempfile
    for combo in combos:
        identity = str((output / combo.run_name()).resolve()).encode()
        lock = Path(tempfile.gettempdir()) / ("auc-frontier-" + hashlib.sha256(identity).hexdigest() + ".lock")
        with lock.open("w") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            run(combo)


def execute_trajectory(
    combo, *, chosen, manifest: dict, output: Path, max_parallel_clients: int,
    partition_summary: Callable[[], dict],
    make_shadows: Callable[[int], tuple],
    eval_seed: int,
    shadow_fingerprints: Callable[[dict], dict] | None = None,
):
    """Train ``combo`` and measure every target in ``chosen`` at every round.

    ``make_shadows(target)`` returns the (clean, noisy) ShadowDataModule pair.
    ``eval_seed`` seeds the evaluation loaders (shadow subsets, server test split).
    ``shadow_fingerprints(shadows)``, if given, is written to ``shadows.json`` so
    IN and OUT runs can be checked to score identical records.
    """
    # Imports are delayed so planning cannot load data or initialize training.
    from experiments.cia import cia
    from experiments.cia.iter_combos import iter_combos
    from metricdp_pytorch.utils.device import resolve_device

    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device()
    folder = output / combo.run_name()
    folder.mkdir(parents=True, exist_ok=True)
    manifest_path = folder / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(f"Manifest mismatch: {folder}; use a separate output directory")
    atomic_json(manifest_path, manifest)
    report = folder / "measurements.json"
    rounds = tuple(range(1, combo.hyperparams.rounds + 1))
    expected = {(r, t) for r in rounds for t in chosen}
    rows = json.loads(report.read_text()) if report.exists() else []
    if (folder / "complete.json").exists() and {(r["round"], r["target"]) for r in rows} == expected:
        return
    atomic_json(folder / "partitions.json", partition_summary())
    import os
    import subprocess
    import sys
    revision = os.environ.get("METRICDP_SOURCE_COMMIT")
    if revision is None:
        revision = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
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
        force=True, log=print, checkpoint_rounds=rounds,
    ):
        if not success:
            raise RuntimeError(f"Training failed: {combo.run_name()}")
        trained = time.monotonic()
        print(f"[FRONTIER] training finished in {trained - started:.1f}s; "
              f"evaluating {len(chosen)} target(s) x {len(rounds)} rounds", flush=True)
        eval_combo = replace(combo, seed=eval_seed)
        shadows = {target: make_shadows(target) for target in chosen}
        if shadow_fingerprints is not None:
            atomic_json(folder / "shadows.json", shadow_fingerprints(shadows))
        rows = []
        for round_number, path in zip(rounds, paths, strict=True):
            for target, (clean, noisy) in shadows.items():
                aggregate, clean_loss, noisy_loss, size = cia.eval_model(
                    path, clean_data_module=clean, noisy_data_module=noisy,
                    device=device, combo=eval_combo,
                )
                rows.append(dict(round=round_number, target=target,
                                 aggregate_loss=aggregate, clean_loss=clean_loss,
                                 noisy_loss=noisy_loss, shadow_size=size))
            atomic_json(report, rows)
            path.unlink()  # only after ALL targets at this round are persisted
            print(f"[FRONTIER EVAL {round_number}/{len(rounds)}] targets={len(chosen)} "
                  f"elapsed={time.monotonic() - trained:.1f}s", flush=True)
        finished = time.monotonic()
        atomic_json(folder / "complete.json", {
            "complete": True, "training_seconds": round(trained - started, 1),
            "evaluation_seconds": round(finished - trained, 1),
        })
