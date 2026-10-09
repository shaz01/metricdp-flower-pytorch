"""Utility-leakage frontier with a practical model-only attack (protocols/2026-10-09_model_based_cia_frontier.md)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

RISK_LABELS = {"q55": 0.55, "q65": 0.65, "q80": 0.8, "q90": 0.9, "q95": 0.95, "nf": None}
TARGETS = (0, 1, 2, 3)
STATISTICS = ("calibrated_records", "raw_records", "class_mix")
BOOTSTRAPS = 2000


def statistic(arrays, name: str, target: int) -> np.ndarray:
    records, mix = arrays["attack_records_gain"][:, target].astype(float), arrays["attack_gain"][:, target].astype(float)
    return {"calibrated_records": records - mix, "raw_records": records, "class_mix": mix}[name]


def auc(in_scores: np.ndarray, out_scores: np.ndarray) -> float:
    """AUC of IN versus OUT scores. A single deterministic release per world (noise-free) is scored 1 / 0.5 / 0 by ordering."""
    if len(in_scores) == 1 and len(out_scores) == 1:
        return 1.0 if in_scores[0] > out_scores[0] else (0.5 if in_scores[0] == out_scores[0] else 0.0)
    labels = np.r_[np.ones(len(in_scores)), np.zeros(len(out_scores))]
    return float(roc_auc_score(labels, np.r_[in_scores, out_scores]))


def world_path(directory: Path, task: str, public_set: int, label: str, world: str) -> Path:
    stem = "frontiernf" if label == "nf" else "frontier"
    risk = "q65" if label == "nf" else label
    return Path(directory) / f"{task}_b32_s{public_set}_A_{risk}_{stem}{world}.json"


def collect(directory: Path, tasks: tuple[str, ...], sets: tuple[int, ...]) -> list[dict]:
    rows = []
    for task in tasks:
        for label in RISK_LABELS:
            for public_set in sets:
                in_path = world_path(directory, task, public_set, label, "in")
                if not in_path.exists():
                    continue
                in_result, in_arrays = json.loads(in_path.read_text(encoding="utf-8")), np.load(in_path.with_suffix(".releases.npz"))
                for target in TARGETS:
                    out_path = world_path(directory, task, public_set, label, f"out{target}")
                    if not out_path.exists():
                        continue
                    out_arrays = np.load(out_path.with_suffix(".releases.npz"))
                    rows.append({"task": task, "label": label, "set": public_set, "target": target,
                                 "auc": {s: auc(statistic(in_arrays, s, target), statistic(out_arrays, s, target)) for s in STATISTICS},
                                 "gain": float(in_result["summary"]["gain_over_control"]), "accuracy_delta": float(in_result["summary"]["accuracy_delta"])})
    return rows


def bootstrap_mean(values_by_set: dict[int, list[float]], seed: int = 0) -> tuple[float, float, float]:
    sets = sorted(values_by_set)
    per_set = np.array([np.mean(values_by_set[s]) for s in sets])
    rng = np.random.default_rng(seed)
    draws = np.array([per_set[rng.integers(0, len(per_set), len(per_set))].mean() for _ in range(BOOTSTRAPS)])
    return float(per_set.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def report(rows: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["label"])].append(row)
    out = {"groups": []}
    for (task, label), members in sorted(groups.items(), key=lambda kv: (kv[0][0], list(RISK_LABELS).index(kv[0][1]))):
        target_risk = RISK_LABELS[label]
        entry = {"task": task, "label": label, "target_risk": target_risk, "n_pairs": len(members), "statistics": {}}
        for name in STATISTICS:
            by_set: dict[int, list[float]] = defaultdict(list)
            for m in members:
                by_set[m["set"]].append(m["auc"][name])
            mean, low, high = bootstrap_mean(by_set)
            entry["statistics"][name] = {"mean_auc": mean, "ci95": [low, high], "detectable": bool(low > 0.5),
                                         "below_target": bool(high < target_risk) if target_risk is not None else None}
        gains_by_set: dict[int, list[float]] = defaultdict(list)
        for m in members:
            gains_by_set[m["set"]].append(m["gain"])
        gain_mean, gain_low, gain_high = bootstrap_mean(gains_by_set)
        entry["utility"] = {"mean_gain": gain_mean, "ci95": [gain_low, gain_high], "mean_accuracy_delta": float(np.mean([m["accuracy_delta"] for m in members]))}
        out["groups"].append(entry)
    primary = [g for g in out["groups"] if g["target_risk"] is not None]
    out["claim_calibration_conservative_for_practical_attack"] = bool(primary and all(g["statistics"]["calibrated_records"]["below_target"] for g in primary))
    out["levels_detectable_by_practical_attack"] = [f"{g['task']}:{g['label']}" for g in out["groups"] if g["statistics"]["calibrated_records"]["detectable"]]
    return out


def format_report(rep: dict) -> str:
    lines = ["Utility-leakage frontier (practical model-only attack; primary = calibrated own-records; 256 releases per world; noise-free = single deterministic release)"]
    for g in rep["groups"]:
        s = g["statistics"]
        target = "n/a" if g["target_risk"] is None else f"{g['target_risk']:.2f}"
        lines.append(f"  {g['task']:<20} {g['label']:<4} target {target}  AUC calibrated {s['calibrated_records']['mean_auc']:.3f} [{s['calibrated_records']['ci95'][0]:.3f},{s['calibrated_records']['ci95'][1]:.3f}]"
                     f"{' DETECTABLE' if s['calibrated_records']['detectable'] else ''}  raw-records {s['raw_records']['mean_auc']:.3f}  class-mix {s['class_mix']['mean_auc']:.3f}"
                     f"  | utility gain {g['utility']['mean_gain']:+.4f} [{g['utility']['ci95'][0]:+.4f},{g['utility']['ci95'][1]:+.4f}], accuracy delta {g['utility']['mean_accuracy_delta']:+.4f}")
    lines.append(f"  claim 'calibrated target upper-bounds the practical attack at every risk': {rep['claim_calibration_conservative_for_practical_attack']}")
    lines.append(f"  levels detectable by the practical attack: {', '.join(rep['levels_detectable_by_practical_attack']) or 'none'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/stacked_head"))
    parser.add_argument("--tasks", nargs="+", default=["kmnist_classes0to3", "kmnist_classes4to7"])
    parser.add_argument("--sets", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    rep = report(collect(args.directory, tuple(args.tasks), tuple(args.sets)))
    text = format_report(rep)
    (args.directory / "frontier_report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
    (args.directory / "frontier_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
