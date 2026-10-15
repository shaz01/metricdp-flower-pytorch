"""WP1 analysis: noise-level, increment and loss attackers on the audit trajectories; gate G1.

Every definition here is fixed in PROTOCOL.md before any protocol run. Trials: for target
``t`` and seed ``s`` the attacker observes either the IN trajectory of seed ``s`` (positive)
or the OUT-``t`` trajectory of seed ``s`` (negative). References are the same cell's
trajectories of the *other* seeds (leave-one-seed-out), so no observed trajectory is ever
its own reference. Scores are computed for every round and pooled over rounds; a cluster
is one target-seed pair (its IN and OUT trials, all rounds).

Usage: ``python -m results.cia_frontier.side_channel_audit.analyze <results root> [--out file]``
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np

from experiments.cia.increment_attack import (
    blind_sigma,
    increment_scores,
    loss_scores,
    noise_level_scores,
    pooled,
    roc_auc,
    stratified_cluster_bootstrap,
    tpr_at_fpr,
)
from results.cia_frontier.eurosat_frontier.runner import EUROSAT_DATA_MODULE

ATTACKERS = ("noise_level", "oracle_noise_level", "increment", "loss")
VIEWS = ("clean", "noisy")
PRIMARY_VIEW = "clean"
G1A_THRESHOLD = 0.55
RESAMPLES = 2000
DATASET_NAMES = {EUROSAT_DATA_MODULE: "eurosat",
                 "results.cia_frontier.side_channel_audit.data:create_data_module": "alzheimer"}


def _metric(name):
    if name == "auc":
        return roc_auc
    fpr = {"tpr_at_5pct_fpr": 0.05, "tpr_at_1pct_fpr": 0.01}[name]
    return lambda positive, negative: tpr_at_fpr(positive, negative, fpr)


METRICS = ("auc", "tpr_at_5pct_fpr", "tpr_at_1pct_fpr")


def load(root):
    """``{cell: {"in": {seed: traj}, "out": {(seed, target): traj}, ...}}`` from complete trajectories."""
    cells = defaultdict(lambda: {"in": {}, "out": {}})
    for manifest_path in sorted(Path(root).rglob("manifest.json")):
        folder = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("diagnostics"):
            raise ValueError(f"Not a diagnostics trajectory: {folder}")
        if not (folder / "complete.json").exists():
            raise ValueError(f"Incomplete trajectory: {folder}")
        rounds = manifest["rounds"]
        rows = json.loads((folder / "measurements.json").read_text())
        expected = {(r, t) for r in range(1, rounds + 1) for t in manifest["targets"]}
        if len(rows) != len(expected) or {(r["round"], r["target"]) for r in rows} != expected:
            raise ValueError(f"Missing or duplicate round/target rows: {folder}")
        releases = sorted(json.loads((folder / "releases.json").read_text()), key=lambda r: r["round"])
        if [r["round"] for r in releases] != list(range(1, rounds + 1)):
            raise ValueError(f"Missing release norms: {folder}")
        run = json.loads((folder / f"{manifest['run_name']}.json").read_text())
        train = run["train_metrics"]
        key = "metric-dp-noise-stdv" if manifest["privacy"] == "metric-privacy" else "global-dp-noise-stdv"
        series = {}
        for target in manifest["targets"]:
            mine = sorted((r for r in rows if r["target"] == target), key=lambda r: r["round"])
            series[target] = {f"{view}_{field}": np.array([r[f"{view}_{field}"] for r in mine])
                              for view in VIEWS for field in ("loss", "increment", "grad_norm_start")}
        trajectory = dict(
            series=series,
            update_norms=np.array([r["update_norm"] for r in releases]),
            parameter_count=releases[0]["parameter_count"],
            true_sigma=np.array([float(train[str(r)][key]) for r in range(1, rounds + 1)]),
            accuracy=float(run["server_evaluate_metrics"][str(rounds)]["accuracy"]),
            records=[c["records"] for c in json.loads((folder / "partitions.json").read_text())["clients"]],
        )
        dataset = DATASET_NAMES[manifest.get("data_module", EUROSAT_DATA_MODULE)]
        cell = cells[(dataset, manifest["clients"], manifest["privacy"])]
        cell.setdefault("rounds", rounds)
        if cell["rounds"] != rounds:
            raise ValueError(f"Mixed round counts in one cell: {folder}")
        identity = manifest["seed"] if manifest["out_target"] is None else (manifest["seed"], manifest["out_target"])
        side = cell["in"] if manifest["out_target"] is None else cell["out"]
        if identity in side:
            raise ValueError(f"Duplicate trajectory: {folder}")
        side[identity] = trajectory
    for key, cell in cells.items():
        seeds = sorted(cell["in"])
        clients = key[1]
        if len(seeds) < 2:
            raise ValueError(f"{key}: leave-one-seed-out references need at least two seeds")
        if set(cell["out"]) != {(s, t) for s in seeds for t in range(clients)}:
            raise ValueError(f"{key}: unbalanced seed x target design")
    return dict(cells)


def top_decile(records):
    """Targets in the top decile by local dataset size (at least one)."""
    count = max(1, math.ceil(0.1 * len(records)))
    return set(np.argsort(records, kind="stable")[::-1][:count].tolist())


def trial_scores(cell, seed, target, world, view=PRIMARY_VIEW):
    """Per-round scores of every attacker for one observed trajectory."""
    observed = cell["in"][seed] if world == "in" else cell["out"][(seed, target)]
    others = [s for s in sorted(cell["in"]) if s != seed]
    ref_in = [cell["in"][s] for s in others]
    ref_out = [cell["out"][(s, target)] for s in others]
    sigma = blind_sigma(observed["update_norms"], observed["parameter_count"])
    out_sigma = np.array([blind_sigma(r["update_norms"], r["parameter_count"]) for r in ref_out])
    field = lambda trajectory, name: trajectory["series"][target][f"{view}_{name}"]
    return {
        "noise_level": noise_level_scores(sigma, out_sigma),
        "oracle_noise_level": noise_level_scores(observed["true_sigma"],
                                                 np.array([r["true_sigma"] for r in ref_out])),
        "increment": increment_scores(
            field(observed, "increment"), field(observed, "grad_norm_start"), sigma,
            np.array([field(r, "increment") for r in ref_in]),
            np.array([field(r, "increment") for r in ref_out]),
        ),
        "loss": loss_scores(field(observed, "loss"), np.array([field(r, "loss") for r in ref_in + ref_out])),
    }


def clusters(cell, view=PRIMARY_VIEW, subgroup="all"):
    """``[(target, seed, {attacker: (positive, negative)})]`` for the requested subgroup."""
    output = []
    for seed in sorted(cell["in"]):
        chosen = top_decile(cell["in"][seed]["records"]) if subgroup == "top_decile" else None
        for target in range(len(cell["in"][seed]["records"])):
            if chosen is not None and target not in chosen:
                continue
            positive = trial_scores(cell, seed, target, "in", view)
            negative = trial_scores(cell, seed, target, "out", view)
            output.append((target, seed, {a: (positive[a], negative[a]) for a in ATTACKERS}))
    return output


def _summary(strata, attacker, metric):
    statistic = pooled(_metric(metric))
    selected = [[c[2][attacker] for c in stratum] for stratum in strata]
    point, interval = stratified_cluster_bootstrap(selected, statistic, resamples=RESAMPLES)
    return {"value": point, "ci95": list(interval)}


def gate_decision(low, high, threshold):
    if low > threshold:
        return "pass"
    if high <= threshold:
        return "fail"
    return "inconclusive"


def side_channel_attribution(oracle_metric_ci_low, blind_metric, blind_global_dp):
    """PROTOCOL.md pre-run revision: which channel a G1(a) pass is attributed to."""
    if oracle_metric_ci_low > G1A_THRESHOLD and blind_metric > blind_global_dp:
        return "metric-privacy noise level"
    return "update norm (both mechanisms)"


def analyze(root):
    cells = load(root)
    report = {"cells": [], "gate": {}}
    for key in sorted(cells):
        cell = cells[key]
        entry = {"dataset": key[0], "clients": key[1], "privacy": key[2], "rounds": cell["rounds"],
                 "seeds": sorted(cell["in"]),
                 "in_accuracy": [cell["in"][s]["accuracy"] for s in sorted(cell["in"])],
                 "out_accuracy_mean": float(np.mean([t["accuracy"] for t in cell["out"].values()])),
                 "attacks": {}}
        for view in VIEWS:
            for subgroup in ("all", "top_decile"):
                strata = [clusters(cell, view, subgroup)]
                entry["attacks"][f"{view}/{subgroup}"] = {
                    attacker: {metric: _summary(strata, attacker, metric) for metric in METRICS}
                    for attacker in ATTACKERS
                }
        report["cells"].append(entry)

    # G1(a): metric privacy, top-decile targets, blind noise-level attack, pooled AUC over all
    # 3-8 client cells of both datasets (bootstrap stratified by cell).
    strata = [clusters(cells[k], PRIMARY_VIEW, "top_decile") for k in sorted(cells) if k[2] == "metric-privacy"]
    if strata:
        g1a = _summary(strata, "noise_level", "auc")
        g1a["decision"] = gate_decision(*g1a["ci95"], G1A_THRESHOLD)
        oracle = _summary(strata, "oracle_noise_level", "auc")
        control = [clusters(cells[k], PRIMARY_VIEW, "top_decile") for k in sorted(cells) if k[2] == "global-dp"]
        blind_global_dp = _summary(control, "noise_level", "auc") if control else None
        g1a["oracle_metric_privacy_auc"] = oracle
        g1a["blind_global_dp_auc"] = blind_global_dp
        if blind_global_dp is not None:
            g1a["attribution"] = side_channel_attribution(
                oracle["ci95"][0], g1a["value"], blind_global_dp["value"])
        report["gate"]["G1a"] = g1a

    # G1(b): macro average over all cells of TPR at 5% FPR, increment minus loss attacker,
    # all targets, clean view (bootstrap stratified by cell, same resample for both attackers).
    strata = [clusters(cells[k], PRIMARY_VIEW, "all") for k in sorted(cells)]
    metric = _metric("tpr_at_5pct_fpr")

    def macro_difference(selected):
        values = []
        for stratum in selected:
            for attacker, sign in (("increment", 1.0), ("loss", -1.0)):
                positive = np.concatenate([c[2][attacker][0] for c in stratum])
                negative = np.concatenate([c[2][attacker][1] for c in stratum])
                values.append(sign * metric(positive, negative))
        return float(np.sum(values) / len(selected))

    if strata:
        point, interval = stratified_cluster_bootstrap(strata, macro_difference, resamples=RESAMPLES)
        report["gate"]["G1b"] = {"value": point, "ci95": list(interval),
                                 "decision": gate_decision(*interval, 0.0)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = json.dumps(analyze(args.root), indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
