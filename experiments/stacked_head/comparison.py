"""Part 1 (paired utility at matched risk) and Part 2 (utility versus practical-attack AUC) of
protocols/2026-10-09_baseline_comparison.md."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.stacked_head import frontier
from experiments.stacked_head.analysis import load_cells

OURS_TAGS = ("confirmsweep", "confirmtransfer")
BASELINE_TAGS = {"global-dp": ("gdpconfirmsweep", "gdpconfirmtransfer"), "metric-privacy": ("mdpconfirmsweep", "mdpconfirmtransfer"), "vanilla": ("vanconfirmsweep", "vanconfirmtransfer")}
VERDICT_GROUPS = 8
BOOTSTRAPS = 2000


def gains(directory: Path, tags: tuple[str, ...], fixed: bool = False) -> dict[tuple[str, float, int], dict]:
    out = {}
    for tag in tags:
        for result in load_cells(directory, tag):
            config = result["config"]
            block = result["summary_fixed_multiplier_1"] if fixed else result["summary"]
            if block is None:
                continue
            oracle = result.get("oracle") or {}
            out[(config["task"], float(config["risk"]), int(config["public-set"]))] = {
                "gain": float(block["gain_over_control"]), "accuracy_delta": float(block["accuracy_delta"]),
                "oracle_auc": float(np.mean(oracle["per_client_mean_oracle_auc"])) if oracle.get("per_client_mean_oracle_auc") else None,
                "fraction_clipped": oracle.get("fraction_clipped"),
            }
    return out


def paired_bootstrap(differences: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = np.array([differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(BOOTSTRAPS)])
    return float(differences.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def classify(mean: float, low: float, high: float) -> str:
    return "ours_better" if mean > 0 and low > 0 else ("baseline_better" if mean < 0 and high < 0 else "no_clear_difference")


def part1(directory: Path) -> dict:
    ours = gains(directory, OURS_TAGS)
    ours_fixed = gains(directory, OURS_TAGS, fixed=True)
    vanilla = gains(directory, BASELINE_TAGS["vanilla"])
    report: dict = {"baselines": {}}
    for mechanism in ("global-dp", "metric-privacy"):
        base = gains(directory, BASELINE_TAGS[mechanism])
        base_fixed = gains(directory, BASELINE_TAGS[mechanism], fixed=True)
        groups = []
        keys = sorted({(t, r) for (t, r, _) in base})
        for task, risk in keys:
            sets = sorted(s for (t, r, s) in base if t == task and r == risk and (t, r, s) in ours)
            if not sets:
                continue
            d = np.array([ours[(task, risk, s)]["gain"] - base[(task, risk, s)]["gain"] for s in sets])
            mean, low, high = paired_bootstrap(d)
            van = [vanilla[(task, 0.65, s)]["gain"] for s in sets if (task, 0.65, s) in vanilla]
            oracle = [base[(task, risk, s)]["oracle_auc"] for s in sets if base[(task, risk, s)]["oracle_auc"] is not None]
            groups.append({
                "task": task, "risk": risk, "n_sets": len(sets), "mean_difference": mean, "ci95": [low, high], "verdict": classify(mean, low, high),
                "ours_mean_gain": float(np.mean([ours[(task, risk, s)]["gain"] for s in sets])), "baseline_mean_gain": float(np.mean([base[(task, risk, s)]["gain"] for s in sets])),
                "baseline_ungated_mean_gain": float(np.mean([base_fixed[(task, risk, s)]["gain"] for s in sets])), "ours_ungated_mean_gain": float(np.mean([ours_fixed[(task, risk, s)]["gain"] for s in sets])),
                "vanilla_mean_gain": float(np.mean(van)) if van else None,
                "ours_accuracy_delta": float(np.mean([ours[(task, risk, s)]["accuracy_delta"] for s in sets])), "baseline_accuracy_delta": float(np.mean([base[(task, risk, s)]["accuracy_delta"] for s in sets])),
                "baseline_realized_oracle_auc": float(np.mean(oracle)) if oracle else None,
                "baseline_fraction_clipped": float(np.mean([base[(task, risk, s)]["fraction_clipped"] for s in sets])),
            })
        counts = defaultdict(int)
        for g in groups:
            counts[g["verdict"]] += 1
        verdict = "ours_better" if counts["ours_better"] >= VERDICT_GROUPS else ("baseline_better" if counts["baseline_better"] >= VERDICT_GROUPS else "mixed")
        report["baselines"][mechanism] = {"groups": groups, "counts": dict(counts), "n_groups": len(groups), "verdict": verdict}
    return report


def interpolated_gain(baseline_points: list[tuple[float, float]], auc: float) -> tuple[float, bool]:
    """Baseline utility at a given practical AUC by piecewise-linear interpolation over its points; flag when outside its range."""
    pts = sorted(baseline_points)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return float(np.interp(auc, xs, ys)), bool(auc < xs[0] or auc > xs[-1])


def part2(directory: Path, tasks=("kmnist_classes0to3", "kmnist_classes4to7"), sets=(0, 1, 2, 3, 4)) -> dict:
    ours = frontier.report(frontier.collect(directory, tuple(tasks), tuple(sets)))
    out: dict = {"baselines": {}}
    for mechanism in ("global-dp", "metric-privacy"):
        rep = frontier.report(frontier.collect(directory, tuple(tasks), tuple(sets), mechanism=mechanism))
        per_task = {}
        for task in tasks:
            def points(report, labels):
                return {g["label"]: (g["statistics"]["calibrated_records"]["mean_auc"], g["utility"]["mean_gain"]) for g in report["groups"] if g["task"] == task and g["label"] in labels}

            base_points = points(rep, frontier.RISK_LABELS)
            ours_points = points(ours, [l for l in frontier.RISK_LABELS if l != "nf"])
            comparisons = []
            for label, (auc, gain) in sorted(ours_points.items()):
                if len(base_points) < 2:
                    continue
                expected, outside = interpolated_gain(list(base_points.values()), auc)
                comparisons.append({"risk": label, "ours_auc": auc, "ours_gain": gain, "baseline_gain_at_same_auc": expected, "outside_baseline_range": outside, "ours_above": bool(gain > expected)})
            per_task[task] = {"baseline_points": base_points, "comparisons": comparisons, "n_above": int(sum(c["ours_above"] for c in comparisons)), "n_levels": len(comparisons)}
        out["baselines"][mechanism] = {"tasks": per_task, "claim_ours_above_curve": bool(per_task and all(v["n_levels"] >= 5 and v["n_above"] >= 4 for v in per_task.values()))}
    return out


def format_part1(report: dict) -> str:
    lines = ["Part 1: paired utility at matched risk (fresh sets, test-split evaluation, gated V0; D = gain(ours) - gain(baseline) in CE)"]
    for mechanism, block in report["baselines"].items():
        lines.append(f"\n  OURS vs {mechanism}: verdict {block['verdict'].upper()}  ({block['counts']} over {block['n_groups']} groups)")
        for g in block["groups"]:
            lines.append(f"    {g['task']:<20} q{g['risk']:<5} n={g['n_sets']:<2} ours {g['ours_mean_gain']:+.4f}  baseline {g['baseline_mean_gain']:+.4f}  (ungated baseline {g['baseline_ungated_mean_gain']:+.4f})  D {g['mean_difference']:+.4f} [{g['ci95'][0]:+.4f},{g['ci95'][1]:+.4f}] {g['verdict']}"
                         f"  | vanilla {g['vanilla_mean_gain'] if g['vanilla_mean_gain'] is None else format(g['vanilla_mean_gain'], '+.4f')}  realized oracle AUC {g['baseline_realized_oracle_auc'] if g['baseline_realized_oracle_auc'] is None else format(g['baseline_realized_oracle_auc'], '.3f')}  clipped {g['baseline_fraction_clipped']:.2f}  acc delta ours {g['ours_accuracy_delta']:+.4f} / baseline {g['baseline_accuracy_delta']:+.4f}")
    return "\n".join(lines)


def format_part2(report: dict) -> str:
    lines = ["Part 2: utility at matched PRACTICAL attack AUC (calibrated own-records statistic)"]
    for mechanism, block in report["baselines"].items():
        lines.append(f"\n  OURS vs {mechanism}: claim 'ours above the baseline curve at >= 4 of 5 risks in both tasks': {block['claim_ours_above_curve']}")
        for task, v in block["tasks"].items():
            lines.append(f"    {task}: ours above at {v['n_above']}/{v['n_levels']} levels; baseline points " + ", ".join(f"{k}:(AUC {a:.3f}, gain {g:+.4f})" for k, (a, g) in sorted(v["baseline_points"].items())))
            for c in v["comparisons"]:
                lines.append(f"      {c['risk']}: ours AUC {c['ours_auc']:.3f} gain {c['ours_gain']:+.4f} vs baseline-at-same-AUC {c['baseline_gain_at_same_auc']:+.4f}{' (outside baseline AUC range)' if c['outside_baseline_range'] else ''} -> {'above' if c['ours_above'] else 'below'}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("part1", "part2"))
    parser.add_argument("--directory", type=Path, default=Path("results/stacked_head"))
    args = parser.parse_args()
    report = part1(args.directory) if args.part == "part1" else part2(args.directory)
    text = format_part1(report) if args.part == "part1" else format_part2(report)
    (args.directory / f"comparison_{args.part}_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    (args.directory / f"comparison_{args.part}_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
