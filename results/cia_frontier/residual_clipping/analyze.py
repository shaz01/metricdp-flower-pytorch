"""WP6 analysis: scores, accuracy, mechanism diagnostics and gate G6 as PROTOCOL.md Sections 3-4 define them.

    uv run --no-sync python -m results.cia_frontier.residual_clipping.analyze \
        results/cia_frontier/residual_clipping/results/stage_a --out <analysis.json>
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from results.cia_frontier.signature_suppression.analyze import cell_statistics

BASELINE = "plain"
CANDIDATES = ("residual-median", "residual-q90")
ACCURACY_LOSS = 0.02
RESAMPLES = 2000
BOOTSTRAP_SEED = 1729


def load(root):
    runs = defaultdict(dict)
    for manifest_path in sorted(Path(root).rglob("manifest.json")):
        folder = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        config = json.loads((folder / "residual_config.json").read_text())["config"]
        if not (folder / "complete.json").exists():
            raise ValueError(f"Incomplete trajectory: {folder}")
        rows = json.loads((folder / "measurements.json").read_text())
        rounds = manifest["rounds"]
        expected = {(r, t) for r in range(1, rounds + 1) for t in manifest["targets"]}
        if len(rows) != len(expected) or {(r["round"], r["target"]) for r in rows} != expected:
            raise ValueError(f"Missing or duplicate round/target: {folder}")
        result = json.loads((folder / f"{manifest['run_name']}.json").read_text())
        train = [result["train_metrics"][str(r)] for r in range(1, rounds + 1)]
        identity = (manifest["seed"], manifest["out_target"])
        if identity in runs[config]:
            raise ValueError(f"Duplicate trajectory: {config} {identity}")
        runs[config][identity] = dict(
            rows=rows, rounds=rounds, targets=manifest["targets"],
            accuracy=float(result["server_evaluate_metrics"][str(rounds)]["accuracy"]),
            rho=[float(m["residual-rho"]) for m in train],
            noise=[float(m["residual-noise-stdv"]) for m in train],
            clipped=[float(m["residual-fraction-clipped"]) for m in train],
            residual_median=[float(np.median(m["residual-norms"])) for m in train],
        )
    data = {}
    for config, trajectories in runs.items():
        seeds = sorted({seed for seed, _ in trajectories})
        cells = {}
        for seed in seeds:
            inside = trajectories[(seed, None)]
            for target in inside["targets"]:
                out = trajectories[(seed, target)]
                losses = {}
                for label, run in (("in", inside), ("out", out)):
                    by_round = {r["round"]: r for r in run["rows"] if r["target"] == target}
                    losses[label] = {view: np.array([by_round[r][f"{view}_loss"] for r in range(1, run["rounds"] + 1)])
                                     for view in ("clean", "noisy")}
                cells[(seed, target)] = {view: losses["out"][view] - losses["in"][view] for view in ("clean", "noisy")}
        values = list(trajectories.values())
        data[config] = dict(
            cells=cells, accuracy=[run["accuracy"] for run in values],
            in_accuracy=[trajectories[(seed, None)]["accuracy"] for seed in seeds],
            rho_median_by_round=np.median([run["rho"] for run in values], axis=0).tolist(),
            noise_median_by_round=np.median([run["noise"] for run in values], axis=0).tolist(),
            clipped_mean_by_round=np.mean([run["clipped"] for run in values], axis=0).tolist(),
            residual_median_by_round=np.median([run["residual_median"] for run in values], axis=0).tolist(),
        )
    return data


def evaluate(data, resamples=RESAMPLES):
    clusters = sorted(data[BASELINE]["cells"])
    configs = [c for c in sorted(data) if sorted(data[c]["cells"]) == clusters]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = [rng.integers(0, len(clusters), len(clusters)) for _ in range(resamples)]
    stats = lambda c, view, chosen: cell_statistics([data[c]["cells"][clusters[i]][view] for i in chosen])
    everything = list(range(len(clusters)))
    point = {c: {v: stats(c, v, everything) for v in ("clean", "noisy")} for c in configs}
    boot = {c: [stats(c, "clean", d) for d in draws] for c in configs}
    interval = lambda xs: [float(x) for x in np.quantile(xs, [0.025, 0.975])]
    accuracy = {c: float(np.mean(data[c]["accuracy"])) for c in configs}
    report = {}
    for c in configs:
        d = data[c]
        report[c] = dict(
            clean=point[c]["clean"], noisy=point[c]["noisy"],
            clean_ci95={k: interval([b[k] for b in boot[c]]) for k in ("score", "auc")},
            per_target_score={f"{s}-{t}": float((d["cells"][(s, t)]["clean"] > 0).mean()) for s, t in clusters},
            accuracy_mean=accuracy[c], accuracy_in=d["in_accuracy"],
            accuracy_range=[min(d["accuracy"]), max(d["accuracy"])], trajectories=len(d["accuracy"]),
            rho_median_rounds_11_on=float(np.median(d["rho_median_by_round"][10:])),
            noise_std_median_rounds_11_on=float(np.median(d["noise_median_by_round"][10:])),
            fraction_clipped_mean_rounds_11_on=float(np.mean(d["clipped_mean_by_round"][10:])),
            residual_norm_median_rounds_11_on=float(np.median(d["residual_median_by_round"][10:])),
            rho_median_by_round=d["rho_median_by_round"], noise_median_by_round=d["noise_median_by_round"],
        )
    gate = {}
    for v in [c for c in CANDIDATES if c in configs]:
        accuracy_ok = accuracy[v] >= accuracy[BASELINE] - ACCURACY_LOSS
        reduction = point[BASELINE]["clean"]["score"] - point[v]["clean"]["score"]
        ci = interval([b["score"] - x["score"] for b, x in zip(boot[BASELINE], boot[v], strict=True)])
        auc = point[BASELINE]["clean"]["auc"] - point[v]["clean"]["auc"]
        auc_ci = interval([b["auc"] - x["auc"] for b, x in zip(boot[BASELINE], boot[v], strict=True)])
        verdict = ("holds" if accuracy_ok and ci[0] > 0 else "fails" if not accuracy_ok or ci[1] <= 0 else "open")
        gate[v] = dict(accuracy=accuracy[v], accuracy_baseline=accuracy[BASELINE], accuracy_condition=accuracy_ok,
                       score_reduction=dict(point=reduction, ci95=ci), auc_reduction=dict(point=auc, ci95=auc_ci),
                       verdict=verdict)
    verdicts = [g["verdict"] for g in gate.values()]
    g6 = ("incomplete (missing configurations; not a gate result)" if set(CANDIDATES + (BASELINE,)) - set(configs)
          else "pass" if "holds" in verdicts else "fail" if all(x == "fails" for x in verdicts) else "inconclusive")
    return dict(clusters=[f"{s}-{t}" for s, t in clusters], baseline=BASELINE, resamples=resamples,
                bootstrap_seed=BOOTSTRAP_SEED, configurations=report,
                gate=dict(G6=g6, candidates=gate,
                          extension_candidates=[v for v, g in gate.items() if g["verdict"] == "open"]
                          if g6 == "inconclusive" else []))


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
