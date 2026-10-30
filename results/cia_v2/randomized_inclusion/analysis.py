"""Load one configuration's randomized-inclusion runs into the arrays of ``experiments.cia.audit_attacks``."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from results.cia_v2.randomized_inclusion.runner import CANDIDATES, TOP_DECILE, design_matrix


@dataclass(frozen=True)
class Audit:
    config: str
    rounds: tuple[int, ...]
    losses: np.ndarray  # (candidates, replicates, checkpoints) shadow loss, in ``candidates`` order
    membership: np.ndarray  # (candidates, replicates) True where the candidate took part
    final_accuracy: np.ndarray  # (replicates,) server test accuracy at the last checkpoint
    released_norms: np.ndarray | None  # (replicates, rounds) released-update norm (DP runs only)
    surrogate_losses: np.ndarray | None  # like ``losses``, label-matched public surrogates
    run_names: tuple[str, ...]
    min_client_accuracy: np.ndarray | None = None  # (replicates,) lowest per-client accuracy at the final round
    candidates: tuple[int, ...] = CANDIDATES  # the run manifests' panel (EuroSAT's by default)
    adjusted_losses: np.ndarray | None = None  # like ``losses``, C1's logit-adjusted shadow loss (logit-adjusted runs)
    top_decile: tuple[int, ...] = TOP_DECILE


def load_audit(root: Path, *, config: str) -> Audit:
    """All runs of ``config`` under ``root``; refuses incomplete or mixed designs."""
    found = []
    for path in sorted(Path(root).glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest["config"] == config:
            found.append((path.parent, manifest))
    if not found:
        raise ValueError(f"no runs of {config!r} under {root}")
    designs = {(m["design_seed"], m["replicates"], tuple(m["checkpoint_rounds"]), tuple(m["candidates"]),
                tuple(m["top_decile"])) for _, m in found}
    if len(designs) != 1:
        raise ValueError(f"{config!r} mixes design settings: {sorted(designs)}")
    ((design_seed, replicates, rounds, candidates, top_decile),) = designs
    by_replicate = {m["replicate"]: (folder, m) for folder, m in found}
    missing = [j for j in range(replicates)
               if j not in by_replicate or not (by_replicate[j][0] / "complete.json").exists()]
    if missing or len(by_replicate) != len(found):
        raise ValueError(f"{config!r} is incomplete or duplicated: replicates {missing} missing")
    expected = design_matrix(replicates, design_seed=design_seed)

    losses = np.empty((len(candidates), replicates, len(rounds)))
    surrogates = adjusted = None
    membership = np.zeros((len(candidates), replicates), dtype=bool)
    accuracy, norms, names = np.empty(replicates), [], []
    client_minimum = np.full(replicates, np.nan)
    for j in range(replicates):
        folder, manifest = by_replicate[j]
        names.append(manifest["run_name"])
        rows = json.loads((folder / "measurements.json").read_text())
        index = {(r["round"], r["candidate"]): r for r in rows}
        for c, candidate in enumerate(candidates):
            membership[c, j] = candidate in manifest["included"]
            for k, round_number in enumerate(rounds):
                row = index[(round_number, candidate)]
                losses[c, j, k] = row["shadow_loss"]
                if "shadow_adjusted_loss" in row:
                    if adjusted is None:
                        adjusted = np.full_like(losses, np.nan)
                    adjusted[c, j, k] = row["shadow_adjusted_loss"]
                if "surrogate_loss" in row:
                    if surrogates is None:
                        surrogates = np.full_like(losses, np.nan)
                    surrogates[c, j, k] = row["surrogate_loss"]
        accuracy[j] = index[(rounds[-1], candidates[0])]["test_accuracy"]
        result = json.loads((folder / f"{manifest['run_name']}.json").read_text())
        training = result["train_metrics"]
        norms.append([training[str(t)].get("dp-released-update-norm", np.nan)
                      for t in sorted(map(int, training))])
        clients = result.get("client_evaluate_metrics", {}).get(str(rounds[-1]), {})
        if "per-client-eval_acc" in clients:
            client_minimum[j] = min(map(float, clients["per-client-eval_acc"]))
    if not np.array_equal(membership, expected):
        raise ValueError(f"{config!r}: recorded inclusion differs from design seed {design_seed}")
    released = np.array(norms, dtype=float)
    return Audit(config=config, rounds=tuple(rounds), losses=losses, membership=membership,
                 final_accuracy=accuracy, released_norms=None if np.isnan(released).all() else released,
                 surrogate_losses=surrogates, run_names=tuple(names),
                 min_client_accuracy=None if np.isnan(client_minimum).all() else client_minimum,
                 candidates=candidates, top_decile=top_decile, adjusted_losses=adjusted)
