"""Shared IN/OUT trajectory execution (used by results/cia_frontier experiments).

Generalized from eurosat_frontier's runner (unchanged behaviour there): train one
trajectory with a checkpoint every round, score every chosen target's clean and
noisy shadow sets on each checkpoint, write ``measurements.json`` atomically and
delete each checkpoint only after all targets at that round are saved.

Layout of one trajectory folder: ``manifest.json`` (immutable plan),
``partitions.json``, ``provenance.json``, the training run JSON,
``measurements.json`` (rows: round, target, aggregate_loss, clean_loss,
noisy_loss, shadow_size), optional ``shadows.json`` (per-target shadow
fingerprints) and ``complete.json``.

Scoring builds the server test loader and every target's clean/noisy shadow loaders
once, then per round loads the checkpoint once and computes the test-set loss once.
Values equal ``experiments.cia.cia.eval_model`` exactly: none of these loaders
shuffle and noisy samples are seeded per record index.
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
        test_loader, shadow_loaders = build_loaders(shadows, eval_combo)
        rows = []
        for round_number, path in zip(rounds, paths, strict=True):
            scores = score_checkpoint(path, test_loader=test_loader, shadow_loaders=shadow_loaders,
                                      device=device, combo=eval_combo)
            for target in shadows:
                aggregate, clean_loss, noisy_loss, size = scores[target]
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


def build_loaders(shadows: dict, combo):
    """Server test loader and per-target (clean, noisy) shadow loaders, as eval_model builds them."""
    batch_size, seed = combo.hyperparams.batch_size, combo.seed
    first_clean = next(iter(shadows.values()))[0]
    _validation, test_loader = first_clean.server_loaders(batch_size=batch_size, seed=seed)
    loaders = {}
    for target, (clean, noisy) in shadows.items():
        clean_loader = clean.target_shadow_loader(batch_size=batch_size, seed=seed)
        noisy_loader = noisy.target_shadow_loader(batch_size=batch_size, seed=seed)
        if len(clean_loader.dataset) != len(noisy_loader.dataset):
            raise ValueError("Clean and noisy shadow datasets must contain the same examples.")
        loaders[target] = (clean_loader, noisy_loader)
    return test_loader, loaders


def score_checkpoint(path, *, test_loader, shadow_loaders: dict, device, combo) -> dict:
    """Return {target: (test loss, clean loss, noisy loss, shadow size)} for one checkpoint."""
    import torch
    from experiments.cia.cia import _calculate_loss
    from metricdp_pytorch.model_module import load_model
    model = load_model(combo.model_module)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    aggregate = _calculate_loss(model, test_loader, device)
    return {target: (aggregate, _calculate_loss(model, clean, device),
                     _calculate_loss(model, noisy, device), len(clean.dataset))
            for target, (clean, noisy) in shadow_loaders.items()}
