"""Step B (public-budget map) and step C (cohort variance decomposition) reports of
protocols/2026-10-09_budgets_and_cohorts.md."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.stacked_head.analysis import distribution, load_cells

BUDGET_RELIABLE_GROUPS = 8
FADE_P_WIN = 0.5
COHORT_RATIO_MAX = 0.10
MAP_SETS = range(10)


def gain_of(result: dict) -> float:
    return float(result["summary"]["gain_over_control"])


def budget_map(directory: Path) -> dict:
    cells = [c for tag in ("sweep", "transfer", "budgetmap") for c in load_cells(directory, tag)]
    groups: dict[tuple[str, float, int], dict[int, float]] = defaultdict(dict)
    for result in cells:
        config = result["config"]
        if config["cohort"] != "A" or int(config["public-set"]) not in MAP_SETS:
            continue
        groups[(config["task"], float(config["risk"]), int(config["budget"]))][int(config["public-set"])] = gain_of(result)
    tasks_risks = sorted({(t, r) for t, r, _ in groups})
    report: dict = {"budgets": {}, "groups": []}
    for budget in sorted({b for _, _, b in groups}):
        per_group, all_gains = [], []
        for task, risk in tasks_risks:
            by_set = groups.get((task, risk, budget), {})
            if len(by_set) < len(MAP_SETS):
                continue
            dist = distribution([by_set[s] for s in sorted(by_set)])
            per_group.append({"task": task, "risk": risk, **dist})
            all_gains.extend(by_set.values())
        if not per_group:
            continue
        gains = np.array(all_gains)
        report["budgets"][str(budget)] = {
            "groups": len(per_group), "groups_reliable": int(sum(g["reliable"] for g in per_group)),
            "reliable": bool(sum(g["reliable"] for g in per_group) >= BUDGET_RELIABLE_GROUPS and len(per_group) >= 10),
            "pooled_p_win": float((gains > 0.001).mean()), "pooled_p_loss": float((gains < -0.003).mean()), "pooled_median_gain": float(np.median(gains)),
            "pooled_mean_gain": float(gains.mean()), "n_cells": int(len(gains)),
        }
        report["groups"].extend({"budget": budget, **g} for g in per_group)
    fade = [int(b) for b, v in report["budgets"].items() if v["pooled_p_win"] < FADE_P_WIN]
    report["fade_budget"] = min(fade) if fade else None
    return report


def anova_components(table: np.ndarray) -> dict[str, float]:
    """Balanced two-way ANOVA without replication; rows are public sets, columns cohorts. Negative components are truncated at 0."""
    n_sets, n_cohorts = table.shape
    grand = table.mean()
    ms_set = n_cohorts * ((table.mean(axis=1) - grand) ** 2).sum() / (n_sets - 1)
    ms_cohort = n_sets * ((table.mean(axis=0) - grand) ** 2).sum() / (n_cohorts - 1)
    residual = table - table.mean(axis=1, keepdims=True) - table.mean(axis=0, keepdims=True) + grand
    ms_resid = (residual**2).sum() / ((n_sets - 1) * (n_cohorts - 1))
    sigma_set, sigma_cohort = max(0.0, (ms_set - ms_resid) / n_cohorts), max(0.0, (ms_cohort - ms_resid) / n_sets)
    total = sigma_set + sigma_cohort + ms_resid
    return {"sigma_set": float(sigma_set), "sigma_cohort": float(sigma_cohort), "sigma_resid": float(ms_resid),
            "share_set": float(sigma_set / total), "share_cohort": float(sigma_cohort / total), "share_resid": float(ms_resid / total),
            "cohort_to_set_ratio": float(sigma_cohort / sigma_set) if sigma_set > 0 else float("inf")}


def cohort_report(directory: Path) -> dict:
    cells = [c for tag in ("sweep", "cohorts") for c in load_cells(directory, tag)]
    tables: dict[tuple[str, float], dict[tuple[int, str], float]] = defaultdict(dict)
    for result in cells:
        config = result["config"]
        if int(config["budget"]) != 32 or int(config["public-set"]) not in MAP_SETS:
            continue
        tables[(config["task"], float(config["risk"]))][(int(config["public-set"]), config["cohort"])] = gain_of(result)
    report: dict = {"groups": []}
    for (task, risk), values in sorted(tables.items()):
        cohorts = sorted({c for _, c in values})
        if len(cohorts) < 4 or len(values) < 40:
            continue
        table = np.array([[values[(s, c)] for c in cohorts] for s in MAP_SETS])
        entry = {"task": task, "risk": risk, "cohorts": cohorts, **anova_components(table), "all_cells": distribution(list(table.ravel()))}
        report["groups"].append(entry)
    report["claim_cohort_small_vs_set"] = bool(report["groups"] and len(report["groups"]) == 4 and all(g["cohort_to_set_ratio"] <= COHORT_RATIO_MAX for g in report["groups"]))
    return report


def format_budget_map(report: dict) -> str:
    lines = ["Public-budget map (cohort A, sets 0-9, 256 releases per cell)"]
    for budget, v in report["budgets"].items():
        lines.append(f"  budget {budget:>3}: groups reliable {v['groups_reliable']}/{v['groups']} -> {'reliable' if v['reliable'] else 'not reliable'}; pooled p_win {v['pooled_p_win']:.2f}, p_loss {v['pooled_p_loss']:.2f}, median gain {v['pooled_median_gain']:+.4f}, mean {v['pooled_mean_gain']:+.4f} (n={v['n_cells']})")
    lines.append(f"  fade budget (pooled p_win < {FADE_P_WIN}): {report['fade_budget']}")
    for g in report["groups"]:
        lines.append(f"    {g['task']:<20} q{g['risk']:<5} b{g['budget']:<4} p_win {g['p_win']:.2f} p_loss {g['p_loss']:.2f} median {g['median_gain']:+.4f} {'reliable' if g['reliable'] else ''}")
    return "\n".join(lines)


def format_cohorts(report: dict) -> str:
    lines = ["Cohort variation (KMNIST, budget 32, sets 0-9 x cohorts A-D)"]
    for g in report["groups"]:
        lines.append(f"  {g['task']:<20} q{g['risk']:<5} shares: set {g['share_set']:.2f}, cohort {g['share_cohort']:.2f}, residual {g['share_resid']:.2f}; cohort/set variance ratio {g['cohort_to_set_ratio']:.3f}; all 40 cells p_win {g['all_cells']['p_win']:.2f} p_loss {g['all_cells']['p_loss']:.2f} median {g['all_cells']['median_gain']:+.4f}")
    lines.append(f"  claim 'cohort variation <= {COHORT_RATIO_MAX:.0%} of public-set variation in all four groups': {report['claim_cohort_small_vs_set']}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which", choices=("budgets", "cohorts"))
    parser.add_argument("--directory", type=Path, default=Path("results/stacked_head"))
    args = parser.parse_args()
    report = budget_map(args.directory) if args.which == "budgets" else cohort_report(args.directory)
    text = format_budget_map(report) if args.which == "budgets" else format_cohorts(report)
    stem = "budget_map_report" if args.which == "budgets" else "cohort_report"
    (args.directory / f"{stem}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (args.directory / f"{stem}.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
