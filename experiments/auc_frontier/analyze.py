"""Final-round analysis only; stored intermediate rounds are never pooled as trials."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np


def statistics(in_loss, out_loss):
    """Matrices are seeds x targets. Lower loss is fixed a priori as more IN-like.

    Paired concordance is not ROC AUC. Target-stratified AUC compares all IN/OUT
    seed pairs for each target, then averages targets without cross-client ranks.
    """
    in_loss, out_loss = np.asarray(in_loss), np.asarray(out_loss)
    if in_loss.shape != out_loss.shape or in_loss.ndim != 2 or not in_loss.size:
        raise ValueError("Require equally shaped nonempty seeds x targets matrices")
    if not (np.isfinite(in_loss).all() and np.isfinite(out_loss).all()):
        raise ValueError("Nonfinite loss")
    def wins(a, b):
        return (a < b).astype(float) + 0.5 * (a == b)
    paired = wins(in_loss, out_loss)
    auc = wins(in_loss[:, None, :], out_loss[None, :, :]).mean()
    return dict(paired_concordance=float(paired.mean()), target_stratified_auc=float(auc),
                per_seed_concordance=paired.mean(axis=1).tolist())


def summarize(in_loss, out_loss, resamples=2000):
    in_loss, out_loss = np.asarray(in_loss), np.asarray(out_loss)
    result = statistics(in_loss, out_loss)
    result.update(seeds=len(in_loss), targets=in_loss.shape[1],
                  uncertainty="Seed-cluster bootstrap, conditional on selected target IDs; few seeds are unstable",
                  ci95=None)
    if len(in_loss) >= 5:
        rng = np.random.default_rng(1729)
        draws = []
        for _ in range(resamples):
            ids = rng.integers(0, len(in_loss), len(in_loss))
            draws.append(statistics(in_loss[ids], out_loss[ids]))
        result["ci95"] = {key: np.quantile([d[key] for d in draws], [0.025, 0.975]).tolist()
                          for key in ("paired_concordance", "target_stratified_auc")}
    return result


def analyze(root, seed_filter=None):
    groups = defaultdict(dict)
    # Collected Colab shards nest trajectories: <root>/<shard>/<run_name>/manifest.json.
    for path in sorted(root.rglob("manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest["pilot"] or (seed_filter is not None and manifest["seed"] not in seed_filter):
            continue
        if not (path.parent / "complete.json").exists():
            raise ValueError(f"Incomplete trajectory: {path.parent}")
        rows = json.loads((path.parent / "measurements.json").read_text())
        expected = {(r, t) for r in range(1, manifest["rounds"] + 1) for t in manifest["targets"]}
        if len(rows) != len(expected) or {(r["round"], r["target"]) for r in rows} != expected:
            raise ValueError(f"Missing or duplicate round/target: {path.parent}")
        losses = {r["target"]: r["clean_loss"] for r in rows if r["round"] == manifest["rounds"]}
        key = (manifest["alpha"], manifest["privacy"], manifest["noise_ratio"], manifest["clients"], manifest["rounds"])
        identity = (manifest["seed"], manifest["out_target"])
        if identity in groups[key]:
            raise ValueError(f"Duplicate trajectory: {identity}")
        run_file = path.parent / (manifest["run_name"] + ".json")
        metrics = json.loads(run_file.read_text())["server_evaluate_metrics"]
        accuracy = float(metrics[str(manifest["rounds"])]["accuracy"])
        groups[key][identity] = (losses, accuracy)
    output = []
    for key, runs in groups.items():
        seeds = sorted({seed for seed, _ in runs})
        if any((seed, None) not in runs for seed in seeds):
            raise ValueError("Missing shared IN trajectory")
        targets = sorted(runs[(seeds[0], None)][0])
        expected = {(s, t) for s in seeds for t in [None, *targets]}
        if set(runs) != expected or any(sorted(runs[(s, None)][0]) != targets for s in seeds):
            raise ValueError("Incomplete or unbalanced seed x target design")
        in_loss = [[runs[(s, None)][0][t] for t in targets] for s in seeds]
        out_loss = [[runs[(s, t)][0][t] for t in targets] for s in seeds]
        output.append(dict(alpha=key[0], privacy=key[1], noise_ratio=key[2], clients=key[3],
                           final_round=key[4], seed_ids=seeds, target_ids=targets,
                           in_accuracy_by_seed=[runs[(s, None)][1] for s in seeds],
                           out_accuracy_by_seed=[[runs[(s, t)][1] for t in targets] for s in seeds],
                           **summarize(in_loss, out_loss)))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", help="Restrict to held-out confirmation seeds")
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.seeds), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
