"""Development-only tuning of the baselines (protocols/2026-10-09_baseline_comparison.md): freeze GDP/VAN local training and
clip norm by public-validation CE, then calibrate the metric-privacy noise multiplier to a target realized oracle AUC."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import ndtr

from experiments.stacked_head.baseline import oracle_auc

TUNING_FILE = "baseline_tuning.json"
RISKS = {"q55": 0.55, "q65": 0.65, "q80": 0.8, "q90": 0.9, "q95": 0.95}
REFERENCE = {"q55": "q65", "q65": "q65", "q80": "q80", "q90": "q65", "q95": "q65"}   # risks without their own tuning reuse the q.65 configuration


def selection_score(result: dict) -> float:
    """Mean public-validation CE of the gated releases of one cell (lower is better)."""
    return float(np.mean([r["validation-ce-chosen"] for r in result["rounds"]]))


def freeze(directory: Path) -> dict:
    gdp: dict[tuple[str, tuple], list[float]] = defaultdict(list)
    van: dict[tuple, list[float]] = defaultdict(list)
    for path in sorted(directory.glob("fmnist_classes0to3_b32_s*_tune*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        base = result["baseline"]
        params = (base["local_lr"], base["local_steps"], base["clip_norm"])
        if base["mechanism"] == "global-dp":
            gdp[(f"q{int(round(float(result['config']['risk']) * 100))}", params)].append(selection_score(result))
        elif base["mechanism"] == "vanilla":
            van[(base["local_lr"], base["local_steps"])].append(selection_score(result))
    out: dict = {"global-dp": {}, "vanilla": {}, "tables": {}, "n_dev_sets": {}}
    for key in ("q65", "q80"):
        table = {params: float(np.mean(scores)) for (k, params), scores in gdp.items() if k == key}
        sizes = {len(scores) for (k, _), scores in gdp.items() if k == key}
        best = min(table, key=table.get)
        out["global-dp"][key] = {"local-lr": best[0], "local-steps": best[1], "clip-norm": best[2], "pooled_validation_ce": table[best]}
        out["tables"][f"global-dp-{key}"] = {f"lr{p[0]}_st{p[1]}_c{p[2]}": v for p, v in sorted(table.items(), key=lambda kv: kv[1])}
        out["n_dev_sets"][key] = sorted(sizes)
    table = {params: float(np.mean(scores)) for params, scores in van.items()}
    best = min(table, key=table.get)
    out["vanilla"] = {"local-lr": best[0], "local-steps": best[1], "pooled_validation_ce": table[best]}
    out["tables"]["vanilla"] = {f"lr{p[0]}_st{p[1]}": v for p, v in sorted(table.items(), key=lambda kv: kv[1])}
    (directory / TUNING_FILE).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def mean_oracle_auc(norms: np.ndarray, distance: np.ndarray, clip: float, nm: float, clients: int = 8) -> float:
    sigma = nm * clip / (distance * clients)          # per release; sigma = nm / d * C / n
    return float(np.mean([oracle_auc(n, clip, s, clients).mean() for n, s in zip(norms, sigma)]))


def solve_noise_multiplier(norms: np.ndarray, distance: np.ndarray, clip: float, risk: float) -> float:
    low, high = 1e-6, 1e4
    for _ in range(200):
        mid = np.sqrt(low * high)
        if mean_oracle_auc(norms, distance, clip, mid) > risk:
            low = mid
        else:
            high = mid
    return float(np.sqrt(low * high))


def calibrate(directory: Path) -> dict:
    tuning = json.loads((directory / TUNING_FILE).read_text(encoding="utf-8"))
    reference_runs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for path in sorted(directory.glob("fmnist_classes0to3_b32_s*_tunem_*.json")):
        ref = path.stem.split("_tunem_")[1]
        arrays = np.load(path.with_suffix(".releases.npz"))
        reference_runs[ref].append((arrays["update_norms"].astype(float), arrays["distance"].astype(float)))
    tuning["metric-privacy"] = {}
    for label, risk in RISKS.items():
        ref = REFERENCE[label]
        config = tuning["global-dp"][ref]
        norms = np.concatenate([n for n, _ in reference_runs[ref]])
        distance = np.concatenate([d for _, d in reference_runs[ref]])
        nm = solve_noise_multiplier(norms, distance, config["clip-norm"], risk)
        tuning["metric-privacy"][label] = {"local-lr": config["local-lr"], "local-steps": config["local-steps"], "clip-norm": config["clip-norm"], "noise-multiplier": nm,
                                           "reference": ref, "dev_mean_oracle_auc": mean_oracle_auc(norms, distance, config["clip-norm"], nm), "dev_mean_distance": float(distance.mean())}
    (directory / TUNING_FILE).write_text(json.dumps(tuning, indent=1) + "\n", encoding="utf-8")
    return tuning


def tuned_parameters(path: Path, mechanism: str, risk: float) -> dict:
    """Frozen parameters for one (mechanism, risk) as runner overrides."""
    tuning = json.loads(Path(path).read_text(encoding="utf-8"))
    label = f"q{int(round(risk * 100))}"
    if mechanism == "vanilla":
        p = tuning["vanilla"]
        return {"local-lr": p["local-lr"], "local-steps": p["local-steps"], "clip-norm": 0.0, "noise-multiplier": None}
    if mechanism == "global-dp":
        p = tuning["global-dp"].get(label) or tuning["global-dp"][REFERENCE[label]]
        return {"local-lr": p["local-lr"], "local-steps": p["local-steps"], "clip-norm": p["clip-norm"], "noise-multiplier": None}
    p = tuning["metric-privacy"][label]
    return {"local-lr": p["local-lr"], "local-steps": p["local-steps"], "clip-norm": p["clip-norm"], "noise-multiplier": p["noise-multiplier"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "calibrate"))
    parser.add_argument("--directory", type=Path, default=Path("results/stacked_head"))
    args = parser.parse_args()
    out = freeze(args.directory) if args.stage == "freeze" else calibrate(args.directory)
    print(json.dumps({k: v for k, v in out.items() if k != "tables"}, indent=1))


if __name__ == "__main__":
    main()
