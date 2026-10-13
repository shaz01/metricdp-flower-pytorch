"""WP2 analysis: scores, AUC, accuracy and gate G2 exactly as PROTOCOL.md Sections 4-5 define them.

Usage (after every trajectory is complete):
    uv run --no-sync python -m results.cia_frontier.signature_suppression.analyze \
        results/cia_frontier/signature_suppression/results/stage_a --out <analysis.json>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

REFERENCE = "vanilla"
CONTROLS = ("weight-decay-5e-4", "local-epochs-1")
CANDIDATES = ("fedprox-mu-0.01", "fedprox-mu-0.1", "fedprox-mu-1", "fixed-steps",
              "consensus-kl-0.3", "consensus-kl-1")
TOP_DECILE = (23, 7)
REDUCTION = 0.05
ACCURACY_LOSS = 0.01
RESAMPLES = 2000
BOOTSTRAP_SEED = 1729
FPR_LEVELS = (0.05, 0.01)


def auc(positive, negative):
    """ROC AUC of ``positive`` over ``negative`` (ties count one half), via average ranks."""
    positive, negative = np.asarray(positive, float), np.asarray(negative, float)
    values = np.concatenate([positive, negative])
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    average_rank = ends - (counts - 1) / 2.0  # 1-based average rank of each distinct value
    rank_sum = average_rank[inverse[: len(positive)]].sum()
    return float((rank_sum - len(positive) * (len(positive) + 1) / 2) / (len(positive) * len(negative)))


def tpr_at_fpr(positive, negative, level):
    """Largest TPR of the rule ``score >= threshold`` whose FPR does not exceed ``level``."""
    positive, negative = np.sort(np.asarray(positive, float)), np.sort(np.asarray(negative, float))
    best = 0.0
    for threshold in np.unique(np.concatenate([positive, negative])):
        fpr = (len(negative) - np.searchsorted(negative, threshold, side="left")) / len(negative)
        if fpr <= level:
            best = max(best, (len(positive) - np.searchsorted(positive, threshold, side="left")) / len(positive))
    return float(best)


def cell_statistics(gaps):
    """``gaps``: list of per-cluster arrays ``L_OUT - L_IN`` over rounds. Section 4 statistics."""
    g = np.concatenate(gaps)
    # Midpoint reference: IN score -(L_IN - m) = g/2, OUT score -(L_OUT - m) = -g/2.
    return dict(score=float((g > 0).mean()), auc=auc(g / 2, -g / 2),
                **{f"tpr_at_{level:g}_fpr": tpr_at_fpr(g / 2, -g / 2, level) for level in FPR_LEVELS})


def load(root):
    """``{config: {"cells": {(seed, target): {view: gap array}}, "accuracy": [...], ...}}``."""
    runs = defaultdict(dict)
    for manifest_path in sorted(Path(root).rglob("manifest.json")):
        folder = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        config = json.loads((folder / "suppression_config.json").read_text())["config"]
        if not (folder / "complete.json").exists():
            raise ValueError(f"Incomplete trajectory: {folder}")
        rows = json.loads((folder / "measurements.json").read_text())
        rounds = manifest["rounds"]
        expected = {(r, t) for r in range(1, rounds + 1) for t in manifest["targets"]}
        if len(rows) != len(expected) or {(r["round"], r["target"]) for r in rows} != expected:
            raise ValueError(f"Missing or duplicate round/target: {folder}")
        result = json.loads((folder / f"{manifest['run_name']}.json").read_text())
        identity = (manifest["seed"], manifest["out_target"])
        if identity in runs[config]:
            raise ValueError(f"Duplicate trajectory: {config} {identity}")
        final = str(rounds)
        clients = result["client_evaluate_metrics"][final]
        loo = defaultdict(list)
        for metrics in result["train_metrics"].values():
            for client, norm in zip(metrics.get("loo-client-ids", []), metrics.get("loo-residual-norms", [])):
                loo[int(client)].append(float(norm))
        runs[config][identity] = dict(
            rows=rows, rounds=rounds, targets=manifest["targets"],
            accuracy=float(result["server_evaluate_metrics"][final]["accuracy"]),
            client_accuracy=dict(zip(map(int, clients["client-ids"]), map(float, clients["per-client-eval_acc"]))),
            loo_median={client: float(np.median(values)) for client, values in loo.items()},
        )
    data = {}
    for config, trajectories in runs.items():
        seeds = sorted({seed for seed, _ in trajectories})
        cells, accuracy = {}, []
        for seed in seeds:
            if (seed, None) not in trajectories:
                raise ValueError(f"Missing IN trajectory: {config} seed {seed}")
            inside = trajectories[(seed, None)]
            for target in inside["targets"]:
                if (seed, target) not in trajectories:
                    raise ValueError(f"Missing OUT trajectory: {config} seed {seed} target {target}")
                out = trajectories[(seed, target)]
                losses = {}
                for label, run in (("in", inside), ("out", out)):
                    by_round = {r["round"]: r for r in run["rows"] if r["target"] == target}
                    losses[label] = {view: np.array([by_round[r][f"{view}_loss"] for r in range(1, run["rounds"] + 1)])
                                     for view in ("clean", "noisy")}
                cells[(seed, target)] = {view: losses["out"][view] - losses["in"][view] for view in ("clean", "noisy")}
            accuracy.extend(run["accuracy"] for (s, _), run in sorted(trajectories.items(), key=str) if s == seed)
        in_runs = [trajectories[(seed, None)] for seed in seeds]
        data[config] = dict(
            cells=cells, accuracy=accuracy,
            in_accuracy=[run["accuracy"] for run in in_runs],
            client_accuracy=[run["client_accuracy"] for run in in_runs],
            loo_median=[{t: run["loo_median"].get(t) for t in run["targets"]} for run in in_runs],
        )
    return data


def evaluate(data, resamples=RESAMPLES):
    configs = sorted(data)
    clusters = sorted(data[REFERENCE]["cells"])
    for config in configs:
        if sorted(data[config]["cells"]) != clusters:
            raise ValueError(f"{config}: clusters differ from {REFERENCE}; comparisons must be paired")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [rng.integers(0, len(clusters), len(clusters)) for _ in range(resamples)]

    def stats(config, view, chosen):
        return cell_statistics([data[config]["cells"][clusters[i]][view] for i in chosen])

    everything = list(range(len(clusters)))
    point = {c: {v: stats(c, v, everything) for v in ("clean", "noisy")} for c in configs}
    boot = {c: [stats(c, "clean", d) for d in draws] for c in configs}
    accuracy = {c: float(np.mean(data[c]["accuracy"])) for c in configs}

    def interval(values):
        return [float(x) for x in np.quantile(values, [0.025, 0.975])]

    report = {}
    for c in configs:
        top = [i for i, (_, t) in enumerate(clusters) if t in TOP_DECILE]
        per_target = {f"{s}-{t}": {v: float((data[c]["cells"][(s, t)][v] > 0).mean()) for v in ("clean", "noisy")}
                      for s, t in clusters}
        report[c] = dict(
            clean=point[c]["clean"], noisy=point[c]["noisy"],
            clean_ci95={k: interval([b[k] for b in boot[c]]) for k in ("score", "auc")},
            top_decile=stats(c, "clean", top) if top else None,
            per_target_score=per_target,
            accuracy_mean=accuracy[c], accuracy_in=data[c]["in_accuracy"],
            accuracy_range=[min(data[c]["accuracy"]), max(data[c]["accuracy"])],
            trajectories=len(data[c]["accuracy"]),
            client_accuracy_in=[dict(min=min(a.values()), mean=float(np.mean(list(a.values()))),
                                     targets={str(t): a.get(t) for t in TOP_DECILE}) for a in data[c]["client_accuracy"]],
            loo_residual_median_in=data[c]["loo_median"],
        )

    def difference(a, b, key):
        """``X(a) - X(b)`` (point, interval) with paired resamples."""
        return (point[a]["clean"][key] - point[b]["clean"][key],
                interval([x[key] - y[key] for x, y in zip(boot[a], boot[b], strict=True)]))

    def matched(control, candidate, key, which):
        """Control's attack level interpolated at the candidate's accuracy (Section 5, part 2)."""
        a0, a1 = accuracy[REFERENCE], accuracy[control]
        x0 = point[REFERENCE]["clean"][key] if which is None else boot[REFERENCE][which][key]
        x1 = point[control]["clean"][key] if which is None else boot[control][which][key]
        if a0 == a1:
            return min(x0, x1)
        weight = min(max((accuracy[candidate] - a0) / (a1 - a0), 0.0), 1.0)
        return x0 + weight * (x1 - x0)

    gate = {}
    for v in [c for c in CANDIDATES if c in data]:
        accuracy_ok = accuracy[v] >= accuracy[REFERENCE] - ACCURACY_LOSS
        reductions = {key: difference(REFERENCE, v, key) for key in ("score", "auc")}
        holds = {key: accuracy_ok and reductions[key][1][0] > REDUCTION for key in reductions}
        part1 = ("holds" if any(holds.values())
                 else "fails" if not accuracy_ok or all(r[1][1] <= REDUCTION for r in reductions.values())
                 else "open")
        key = "score" if holds["score"] or not holds["auc"] else "auc"
        beats = {}
        for control in [c for c in CONTROLS if c in data]:
            values = [matched(control, v, key, i) - boot[v][i][key] for i in range(resamples)]
            beats[control] = dict(point=matched(control, v, key, None) - point[v]["clean"][key],
                                  ci95=interval(values))
        part2 = ("fails" if any(b["ci95"][1] <= 0 for b in beats.values())
                 else "holds" if len(beats) == len(CONTROLS) and all(b["ci95"][0] > 0 for b in beats.values())
                 else "open")
        gate[v] = dict(accuracy=accuracy[v], accuracy_condition=accuracy_ok,
                       reduction_score=dict(point=reductions["score"][0], ci95=reductions["score"][1]),
                       reduction_auc=dict(point=reductions["auc"][0], ci95=reductions["auc"][1]),
                       part1=part1, part2_statistic=key, beats_controls=beats, part2=part2)
    passing = [v for v, g in gate.items() if g["part1"] == "holds" and g["part2"] == "holds"]
    failed = [v for v, g in gate.items() if "fails" in (g["part1"], g["part2"])]
    verdict = "pass" if passing else "fail" if len(failed) == len(gate) else "inconclusive"
    if set(CANDIDATES + CONTROLS + (REFERENCE,)) - set(data):
        verdict = "incomplete (missing configurations; not a gate result)"
    rank = lambda v: gate[v]["reduction_score" if gate[v]["part2_statistic"] == "score" else "reduction_auc"]["point"]
    open_candidates = sorted((v for v in gate if v not in failed), key=lambda v: -gate[v]["reduction_score"]["point"])
    return dict(
        clusters=[f"{s}-{t}" for s, t in clusters], reference=REFERENCE, controls=list(CONTROLS),
        resamples=resamples, bootstrap_seed=BOOTSTRAP_SEED, configurations=report,
        gate=dict(G2=verdict, candidates=gate, passing=passing,
                  best_two=sorted(passing, key=lambda v: -rank(v))[:2],
                  extension_candidates=open_candidates[:2] if verdict == "inconclusive" else []),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    args = parser.parse_args()
    text = json.dumps(evaluate(load(args.root), args.resamples), indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
