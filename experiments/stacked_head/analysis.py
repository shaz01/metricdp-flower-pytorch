"""Summarise finished stacked-head cells and apply the pre-stated gate (per task and risk, budget by budget)."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

GATE_MIN_CELLS = 5
GATE_MIN_GAIN = 0.001
GATE_WORST_CELL = -0.003


def load_cells(directory: Path, tag: str | None = None) -> list[dict]:
    cells = []
    for path in sorted(Path(directory).glob("*.json")):
        if path.name.endswith("_attack.json") or "_attack_" in path.name:
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("summary") is None or "config" not in result:
            continue
        if tag is not None and not path.stem.endswith(f"_{tag}"):
            continue
        cells.append(result)
    return cells


def gate(gains: list[float], cells_expected: int = 6) -> dict:
    """Gate used in the research phase: >= 5/6 cells above .001, mean > 0, no cell below -.003."""
    above = sum(g > GATE_MIN_GAIN for g in gains)
    return {
        "cells": len(gains), "cells_above_threshold": int(above), "mean_gain": float(np.mean(gains)), "worst_cell": float(min(gains)),
        "passed": bool(len(gains) >= cells_expected and above >= GATE_MIN_CELLS and np.mean(gains) > 0 and min(gains) >= GATE_WORST_CELL),
    }


def summarize(cells: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for result in cells:
        config = result["config"]
        groups[(config["task"], int(config["budget"]), float(config["risk"]))].append(result)
    rows = []
    for (task, budget, risk), members in sorted(groups.items()):
        gains = [m["summary"]["gain_over_control"] for m in members]
        accuracy = [m["summary"]["accuracy_delta"] for m in members]
        rows.append({"task": task, "budget": budget, "risk": risk, "mean_accuracy_delta": float(np.mean(accuracy)), **gate(gains)})
    return rows


# ---- public-set sweep / transfer report (protocol 2026-10-09_public_set_sweep_and_transfer.md) -------------------------

P_WIN_MIN = 0.80
P_LOSS_MAX = 0.10
LOSS_THRESHOLD = -0.003
GATES = {"gated_val512": ("summary", None), "gated_val128": ("summary_alt_validation", "128"), "fixed_multiplier_1": ("summary_fixed_multiplier_1", None)}


def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denominator
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def gate_gain(result: dict, gate: str) -> float | None:
    key, sub = GATES[gate]
    block = result.get(key)
    if block is None:
        return None
    if sub is not None:
        block = block.get(sub)
    return None if block is None else float(block["gain_over_control"])


def distribution(gains: list[float]) -> dict:
    gains_array = np.array(gains)
    wins, losses = int((gains_array > GATE_MIN_GAIN).sum()), int((gains_array < LOSS_THRESHOLD).sum())
    n = len(gains_array)
    return {
        "n_sets": n, "p_win": wins / n, "p_win_ci95": wilson(wins, n), "p_loss": losses / n, "p_loss_ci95": wilson(losses, n),
        "median_gain": float(np.median(gains_array)), "p10_gain": float(np.percentile(gains_array, 10)), "mean_gain": float(gains_array.mean()),
        "reliable": bool(wins / n >= P_WIN_MIN and losses / n <= P_LOSS_MAX and np.median(gains_array) > 0),
    }


def failure_anatomy(result: dict, directory: Path) -> dict:
    """For one failing set: would any fixed step multiplier have beaten the control on held-out data? What did the gate pick?"""
    stored = Path(directory) / f"{result['run_name']}.releases.npz"
    pick_fraction = result["summary"]["pick_fraction"]
    out = {"set": int(result["config"]["public-set"]), "gain": gate_gain(result, "gated_val512"), "control_ce": result["control"]["ce"], "pick_fraction": pick_fraction}
    if stored.exists():
        arrays = np.load(stored)
        mean_ce = arrays["eval_ce"].mean(axis=0)
        best = int(mean_ce.argmin())
        out["mean_eval_ce_by_multiplier"] = {str(m): float(c) for m, c in zip(arrays["multipliers"], mean_ce)}
        out["best_fixed_multiplier"] = float(arrays["multipliers"][best])
        out["best_fixed_gain"] = float(result["control"]["ce"] - mean_ce[best])
        out["cause"] = "gate_error" if out["best_fixed_gain"] > GATE_MIN_GAIN else "no_usable_step"
    return out


def sweep_report(directory: Path, tag: str) -> dict:
    cells = load_cells(directory, tag)
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for result in cells:
        groups[(result["config"]["task"], float(result["config"]["risk"]))].append(result)
    report: dict = {"tag": tag, "rule": {"p_win_min": P_WIN_MIN, "p_loss_max": P_LOSS_MAX, "win_threshold": GATE_MIN_GAIN, "loss_threshold": LOSS_THRESHOLD}, "groups": []}
    for (task, risk), members in sorted(groups.items()):
        members.sort(key=lambda r: int(r["config"]["public-set"]))
        entry: dict = {"task": task, "risk": risk, "gates": {}, "sets": [int(m["config"]["public-set"]) for m in members]}
        for gate in GATES:
            pairs = [(m, gate_gain(m, gate)) for m in members]
            entry["gates"][gate] = distribution([g for _, g in pairs if g is not None])
        gains = np.array([gate_gain(m, "gated_val512") for m in members])
        control = np.array([m["control"]["ce"] for m in members])
        if len(members) >= 4:
            rho = spearmanr(control, gains)
            entry["spearman_gain_vs_control_ce"] = {"rho": float(rho.statistic), "p_value": float(rho.pvalue)}
        entry["mean_accuracy_delta"] = float(np.mean([m["summary"]["accuracy_delta"] for m in members]))
        entry["control_ce_range"] = [float(control.min()), float(control.max())]
        entry["failures"] = [failure_anatomy(m, directory) for m, g in zip(members, gains) if g < LOSS_THRESHOLD]
        report["groups"].append(entry)
    return report


def dataset_transfer(report: dict) -> dict:
    """A dataset transfers iff the reliability rule holds for at least 3 of its (task, risk) pairs (all of them when it has only 2)."""
    by_dataset: dict[str, list[bool]] = defaultdict(list)
    for entry in report["groups"]:
        by_dataset[entry["task"].split("_")[0]].append(entry["gates"]["gated_val512"]["reliable"])
    return {name: {"pairs_reliable": int(sum(flags)), "pairs": len(flags), "transfers": bool(sum(flags) >= (3 if len(flags) >= 4 else len(flags)))} for name, flags in by_dataset.items()}


def format_report(report: dict) -> str:
    lines = [f"Public-set report (tag {report['tag']}); rule: p_win>={P_WIN_MIN} (gain>{GATE_MIN_GAIN}), p_loss<={P_LOSS_MAX} (gain<{LOSS_THRESHOLD}), median>0"]
    for entry in report["groups"]:
        lines.append(f"\n{entry['task']}  q{entry['risk']}  sets {entry['sets'][0]}..{entry['sets'][-1]} (n={entry['gates']['gated_val512']['n_sets']})  mean acc delta {entry['mean_accuracy_delta']:+.4f}  control CE {entry['control_ce_range'][0]:.3f}-{entry['control_ce_range'][1]:.3f}")
        for gate, d in entry["gates"].items():
            lines.append(f"  {gate:<20} p_win {d['p_win']:.2f} [{d['p_win_ci95'][0]:.2f},{d['p_win_ci95'][1]:.2f}]  p_loss {d['p_loss']:.2f} [{d['p_loss_ci95'][0]:.2f},{d['p_loss_ci95'][1]:.2f}]  median {d['median_gain']:+.4f}  p10 {d['p10_gain']:+.4f}  mean {d['mean_gain']:+.4f}  {'RELIABLE' if d['reliable'] else 'not reliable'}")
        if "spearman_gain_vs_control_ce" in entry:
            sp = entry["spearman_gain_vs_control_ce"]
            lines.append(f"  Spearman(gain, control CE) rho {sp['rho']:+.2f} (p {sp['p_value']:.3g})")
        for failure in entry["failures"]:
            lines.append(f"  failure: set {failure['set']} gain {failure['gain']:+.4f} cause {failure.get('cause', 'n/a')} best fixed multiplier {failure.get('best_fixed_multiplier', 'n/a')} (gain {failure.get('best_fixed_gain', float('nan')):+.4f})")
    transfer = dataset_transfer(report)
    lines.append("\nDataset-level (512-validation gate): " + "; ".join(f"{k}: {v['pairs_reliable']}/{v['pairs']} reliable -> {'transfers' if v['transfers'] else 'does not transfer'}" for k, v in sorted(transfer.items())))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("results/stacked_head"))
    parser.add_argument("--tag", default=None, help="only result files ending in _<tag>.json")
    parser.add_argument("--sweep-report", action="store_true", help="per-task distribution over public sets (needs --tag sweep or transfer); writes sweep_report_<tag>.{json,txt}")
    args = parser.parse_args()
    if args.sweep_report:
        report = sweep_report(args.directory, args.tag)
        report["dataset_transfer"] = dataset_transfer(report)
        text = format_report(report)
        print(text)
        (args.directory / f"sweep_report_{args.tag}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        (args.directory / f"sweep_report_{args.tag}.txt").write_text(text + "\n", encoding="utf-8")
        return
    rows = summarize(load_cells(args.directory, args.tag))
    for row in rows:
        print(f"{row['task']:<22} b{row['budget']:<4} q{row['risk']:<5} cells {row['cells']}  >.001: {row['cells_above_threshold']}/{row['cells']}  "
              f"mean {row['mean_gain']:+.4f}  worst {row['worst_cell']:+.4f}  acc {row['mean_accuracy_delta']:+.4f}  gate {'PASS' if row['passed'] else 'not met'}")


if __name__ == "__main__":
    main()
