"""A2 analysis: per cell and mechanism, the calibrated loss attacker and the blind noise-level attacker; G-A2.

Trials are the cell's 2R trajectories, in pair order (IN_j, OUT_j). Features:

- loss: the target's clean shadow loss at rounds 0, 5, ..., 50 (round 0 from round 1's ``clean_loss_before``);
- noise level (blind, primary): the mean over rounds of ``sigma_hat_t = ||Delta_t|| / sqrt(D)``, from the logged
  released-update norm and parameter count; the 50-round vector as sensitivity;
- oracle: the strategy's logged true noise std (mean over rounds), threshold AUC only.

Collapsed rounds (amendment, owner decision 2026-10-19; PROTOCOL.md Section 6): when every client update has zero
norm the strategy skips aggregation and releases nothing, so the round's released norm and true noise std are 0.
A pilot level with any collapsed global-DP trajectory does not qualify.

Intervals: ``pair_jackknife`` at ``GATE_LEVEL`` (STATE.md, Attacker and metrics, for its calibration at this size).

    uv run --no-sync python -m results.cia_v2.side_channel.analyze [root] --out results/cia_v2/side_channel/analysis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cia.audit_attacks import GATE_LEVEL, calibrated_scores, pair_jackknife, roc_auc, tpr_at_fpr

HERE = Path(__file__).resolve().parent
CHECKPOINTS = tuple(range(0, 51, 5))
NOISE_KEYS = ("metric-dp-noise-stdv", "global-dp-noise-stdv")
COLLAPSE_KEYS = ("metric-dp-aggregation-collapsed", "global-dp-aggregation-collapsed")
NOISE_LIMIT, LOSS_LIMIT, CELLS_NEEDED = 0.8, 0.6, 2


def _training(folder, manifest):
    return json.loads((folder / f"{manifest['run_name']}.json").read_text())


def load_cell(root):
    """One (cell, mechanism, noise multiple) directory -> paired feature arrays."""
    runs = {}
    for path in sorted(Path(root).glob("*/manifest.json")):
        folder = path.parent
        if not (folder / "complete.json").exists():
            raise ValueError(f"incomplete trajectory: {folder}")
        manifest = json.loads(path.read_text())
        replicate = json.loads((folder / "a2_seeds.json").read_text())["replicate"]
        key = (replicate, manifest["out_target"] is not None)
        if key in runs:
            raise ValueError(f"duplicate trajectory {key} in {root}")
        runs[key] = (folder, manifest)
    replicates = sorted({j for j, _ in runs})
    if replicates != list(range(len(replicates))) or len(runs) != 2 * len(replicates):
        raise ValueError(f"{root}: every replicate needs one IN and one OUT trajectory")
    losses, sigma_hat, sigma_rounds, oracle, accuracy, membership, collapsed = [], [], [], [], [], [], []
    for j in replicates:
        for out in (False, True):
            folder, manifest = runs[(j, out)]
            (target,) = manifest["targets"]
            rows = {r["round"]: r for r in json.loads((folder / "measurements.json").read_text())
                    if r["target"] == target}
            losses.append([rows[1]["clean_loss_before"] if c == 0 else rows[c]["clean_loss"] for c in CHECKPOINTS])
            result = _training(folder, manifest)
            metrics = [result["train_metrics"][str(t)] for t in sorted(map(int, result["train_metrics"]))]
            released = [m for m in metrics if not _collapsed(m)]
            parameters = released[0]["dp-parameter-count"]
            key = next(k for k in NOISE_KEYS if k in released[0])
            per_round = np.array([0.0 if _collapsed(m) else m["dp-released-update-norm"] / np.sqrt(parameters)
                                  for m in metrics])
            sigma_rounds.append(per_round)
            sigma_hat.append([per_round.mean()])
            oracle.append(float(np.mean([0.0 if _collapsed(m) else m[key] for m in metrics])))
            collapsed.append(len(metrics) - len(released))
            final = str(max(map(int, result["server_evaluate_metrics"])))
            accuracy.append(result["server_evaluate_metrics"][final]["accuracy"])
            membership.append(not out)
    return {"losses": np.array(losses)[None], "sigma_hat": np.array(sigma_hat)[None],
            "sigma_rounds": np.array(sigma_rounds)[None], "oracle_sigma": np.array(oracle)[None],
            "membership": np.array(membership)[None], "accuracy": np.array(accuracy),
            "collapsed_rounds": np.array(collapsed)}


def _collapsed(metrics):
    return any(metrics.get(key) == 1.0 for key in COLLAPSE_KEYS)


def _attack(features, membership):
    scores = calibrated_scores(features, membership)
    result = pair_jackknife(scores, membership, roc_auc)
    return {"auc": {"estimate": result.estimate, "low": result.low, "high": result.high},
            "tpr_at_5pct_fpr": tpr_at_fpr(scores, membership, 0.05)}


def cell_summary(data):
    member = data["membership"]
    oracle = roc_auc(data["oracle_sigma"], member)
    return {
        "trajectories": int(member.shape[1]), "gate_level": GATE_LEVEL,
        "loss": _attack(data["losses"], member),
        "loss_final_round": _attack(data["losses"][..., -1:], member),
        "noise_level": _attack(data["sigma_hat"], member),
        "noise_level_rounds": _attack(data["sigma_rounds"], member),
        # The true std may be higher or lower IN than OUT: report the separation either way.
        "oracle_noise_level_auc": max(oracle, 1 - oracle),
        "accuracy_in": float(data["accuracy"][member[0]].mean()),
        "accuracy_out": float(data["accuracy"][~member[0]].mean()),
        "collapsed_trajectories": int((data["collapsed_rounds"] > 0).sum()),
        "collapsed_rounds": int(data["collapsed_rounds"].sum()),
    }


def pilot_decision(loss_auc, multiple, multiples=(1, 4, 16, 64), *, collapsed_trajectories=0):
    """Pilot rule (PROTOCOL.md Section 2): the lowest multiple whose global-DP loss AUC (point) is 0.6 or below.

    A level with a collapsed trajectory does not qualify, and more noise only collapses sooner (Section 6)."""
    if collapsed_trajectories:
        return "no qualifying noise", None
    if loss_auc <= LOSS_LIMIT:
        return "qualifies", multiple
    later = [m for m in multiples if m > multiple]
    return ("next", later[0]) if later else ("no qualifying noise", None)


def g_a2(cells):
    """``cells``: {cell: {mechanism: cell_summary}}. Strong claim: >= 2 cells with both conditions."""
    qualifying = [name for name, result in cells.items()
                  if "metric-privacy" in result and "global-dp" in result
                  and result["metric-privacy"]["noise_level"]["auc"]["low"] >= NOISE_LIMIT
                  and result["global-dp"]["loss"]["auc"]["estimate"] <= LOSS_LIMIT]
    return {"verdict": "strong claim" if len(qualifying) >= CELLS_NEEDED else "channel size recorded",
            "qualifying_cells": qualifying}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=HERE / "results" / "main")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--pilot", action="store_true", help="root is a pilot stage: print the pilot decision per cell")
    args = parser.parse_args()
    if args.pilot:
        for directory in sorted(p for p in args.root.glob("*/clients-*/global-dp-x*") if p.is_dir()):
            if len(list(directory.glob("*/complete.json"))) < 2 * 4:
                print(f"{directory}: incomplete")
                continue
            summary = cell_summary(load_cell(directory))
            multiple = int(directory.name.rsplit("-x", 1)[1])
            print(json.dumps({"cell": str(directory.relative_to(args.root)), "loss_auc": summary["loss"]["auc"],
                              "noise_level_auc": summary["noise_level"]["auc"],
                              "accuracy_in": summary["accuracy_in"],
                              "collapsed_trajectories": summary["collapsed_trajectories"],
                              "decision": pilot_decision(summary["loss"]["auc"]["estimate"], multiple,
                                                         collapsed_trajectories=summary["collapsed_trajectories"])}))
        return
    cells = {}
    for directory in sorted(p for p in args.root.glob("*/clients-*/*-x*") if p.is_dir()):
        dataset, clients, mechanism_level = directory.relative_to(args.root).parts
        mechanism = mechanism_level.rsplit("-x", 1)[0]
        cells.setdefault(f"{dataset}-{clients}", {})[mechanism] = cell_summary(load_cell(directory)) | {
            "noise_multiple": int(mechanism_level.rsplit("-x", 1)[1])}
    report = {"cells": cells, "g_a2": g_a2(cells)}
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
