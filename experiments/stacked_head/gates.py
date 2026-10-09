"""Offline gate variants V0-V3 on stored per-release arrays, with the selection and confirmation rules of
protocols/2026-10-09_accuracy_aware_gate.md.

Every variant only reads the validation scores of the eight step multipliers (public data and the released aggregate),
so all of them are post-processing and share V0's privacy accounting.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.stacked_head.analysis import GATE_MIN_GAIN, LOSS_THRESHOLD, distribution, wilson

VARIANTS = ("V0", "V1", "V2", "V3")
DIP_THRESHOLD = -0.005
MEDIAN_RETENTION = 0.8
DIP_REDUCTION_REQUIRED = 0.30
DEV_TAGS = ("sweep", "transfer")
CONFIRM_TAGS = ("confirmsweep", "confirmtransfer")
RELAXATION = 0.01


def picks(variant: str, val_ce: np.ndarray, val_accuracy: np.ndarray) -> np.ndarray:
    """Multiplier index chosen for each release. Column 0 is the public base (multiplier 0) and always qualifies."""
    if variant == "V0":
        return val_ce.argmin(axis=1)
    if variant in ("V1", "V3"):
        slack = 0.0 if variant == "V1" else RELAXATION
        admissible = val_accuracy >= val_accuracy[:, :1] - slack
    elif variant == "V2":
        admissible = val_ce <= val_ce[:, :1]
        top = np.where(admissible, val_accuracy, -np.inf).max(axis=1, keepdims=True)
        admissible = admissible & (val_accuracy == top)
    else:
        raise ValueError(f"Unknown gate variant {variant!r}.")
    admissible = admissible.copy()
    admissible[:, 0] = admissible[:, 0] | (variant != "V2") | (~admissible.any(axis=1))
    return np.where(admissible, val_ce, np.inf).argmin(axis=1)


def cell_outcomes(result: dict, arrays) -> dict[str, dict[str, float]]:
    control = result["control"]
    eval_ce, eval_accuracy = arrays["eval_ce"].astype(float), arrays["eval_accuracy"].astype(float)
    val_ce, val_accuracy = arrays["val_ce"].astype(float), arrays["val_accuracy"].astype(float)
    rows = np.arange(len(val_ce))
    out = {}
    for variant in VARIANTS:
        choice = picks(variant, val_ce, val_accuracy)
        out[variant] = {
            "gain": float(control["ce"] - eval_ce[rows, choice].mean()),
            "accuracy_delta": float(eval_accuracy[rows, choice].mean() - control["accuracy"]),
            "mean_multiplier_is_base": float((choice == 0).mean()),
        }
    return out


def load_outcomes(directory: Path, tags: tuple[str, ...]) -> list[dict]:
    cells = []
    for tag in tags:
        for path in sorted(Path(directory).glob(f"*_{tag}.json")):
            if path.name.startswith(("sweep_report_", "gate_")):
                continue
            result = json.loads(path.read_text(encoding="utf-8"))
            if "config" not in result or result.get("summary") is None:
                continue
            arrays = np.load(path.with_suffix(".releases.npz"))
            if "val_accuracy" not in arrays.files:
                raise ValueError(f"{path.name} has no stored validation accuracy; re-run the cell with the current runner.")
            cells.append({"task": result["config"]["task"], "risk": float(result["config"]["risk"]), "set": int(result["config"]["public-set"]), "name": path.stem, "outcomes": cell_outcomes(result, arrays)})
    return cells


def report(cells: list[dict]) -> dict:
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for cell in cells:
        groups[(cell["task"], cell["risk"])].append(cell)
    out = {"n_cells": len(cells), "variants": {}}
    for variant in VARIANTS:
        per_group = {}
        for (task, risk), members in sorted(groups.items()):
            gains = [m["outcomes"][variant]["gain"] for m in members]
            per_group[f"{task}|q{risk}"] = distribution(gains)
        dips = [m["outcomes"][variant]["accuracy_delta"] < DIP_THRESHOLD for m in cells]
        dip_share = float(np.mean(dips))
        out["variants"][variant] = {
            "groups": per_group,
            "groups_reliable": int(sum(g["reliable"] for g in per_group.values())),
            "n_groups": len(per_group),
            "accuracy_dip_share": dip_share,
            "accuracy_dip_ci95": wilson(int(sum(dips)), len(dips)),
            "mean_accuracy_delta": float(np.mean([m["outcomes"][variant]["accuracy_delta"] for m in cells])),
            "worst_accuracy_delta": float(min(m["outcomes"][variant]["accuracy_delta"] for m in cells)),
            "share_cells_at_base": float(np.mean([m["outcomes"][variant]["mean_multiplier_is_base"] for m in cells])),
        }
    return out


def select_variant(dev: dict) -> dict:
    """Pre-stated development rule: eligible = reliable in all groups and median >= 0.8 x V0's in every group; adopt the
    eligible variant with the lowest dip share only if it is at most 0.7 x V0's."""
    v0 = dev["variants"]["V0"]
    decisions = {}
    for variant in VARIANTS[1:]:
        v = dev["variants"][variant]
        all_reliable = v["groups_reliable"] == v["n_groups"]
        retention = {k: (g["median_gain"] / v0["groups"][k]["median_gain"]) if v0["groups"][k]["median_gain"] > 0 else float("nan") for k, g in v["groups"].items()}
        keeps_median = all(r >= MEDIAN_RETENTION for r in retention.values())
        decisions[variant] = {"all_groups_reliable": all_reliable, "keeps_80pct_of_v0_median_everywhere": keeps_median, "eligible": bool(all_reliable and keeps_median),
                              "dip_share": v["accuracy_dip_share"], "dip_ratio_to_v0": v["accuracy_dip_share"] / v0["accuracy_dip_share"] if v0["accuracy_dip_share"] > 0 else float("nan"),
                              "min_median_retention": float(np.nanmin(list(retention.values())))}
    eligible = {k: d for k, d in decisions.items() if d["eligible"]}
    chosen, reason = "V0", "no eligible variant"
    if eligible:
        best = min(eligible, key=lambda k: eligible[k]["dip_share"])
        if eligible[best]["dip_share"] <= 0.7 * v0["accuracy_dip_share"]:
            chosen, reason = best, f"{best} has the lowest dip share among eligible variants and is at most 0.7 x V0's"
        else:
            reason = f"best eligible variant {best} reduces the dip share by less than 30% relative to V0"
    return {"decisions": decisions, "chosen": chosen, "reason": reason, "v0_dip_share": v0["accuracy_dip_share"]}


def confirm(fresh: dict, chosen: str) -> dict:
    """Pre-stated confirmation: chosen variant reliable in >= 9 of 10 groups, and (if not V0) dip share at least 30% below V0's."""
    c, v0 = fresh["variants"][chosen], fresh["variants"]["V0"]
    reduction = 1 - c["accuracy_dip_share"] / v0["accuracy_dip_share"] if v0["accuracy_dip_share"] > 0 else float("nan")
    reliable_ok = c["groups_reliable"] >= 9
    dip_ok = True if chosen == "V0" else bool(reduction >= DIP_REDUCTION_REQUIRED)
    return {"chosen": chosen, "groups_reliable": c["groups_reliable"], "n_groups": c["n_groups"], "reliability_ok": bool(reliable_ok), "dip_share_chosen": c["accuracy_dip_share"],
            "dip_share_v0": v0["accuracy_dip_share"], "dip_relative_reduction_vs_v0": float(reduction), "dip_ok": dip_ok, "confirmed": bool(reliable_ok and dip_ok),
            "v0_groups_reliable": v0["groups_reliable"]}


def format_report(rep: dict, title: str) -> str:
    lines = [f"{title}: {rep['n_cells']} cells"]
    for variant, v in rep["variants"].items():
        lo, hi = v["accuracy_dip_ci95"]
        lines.append(f"  {variant}: groups reliable {v['groups_reliable']}/{v['n_groups']}  accuracy-dip share {v['accuracy_dip_share']:.3f} [{lo:.3f},{hi:.3f}]  mean acc delta {v['mean_accuracy_delta']:+.4f}  worst {v['worst_accuracy_delta']:+.4f}  releases at base {v['share_cells_at_base']:.3f}")
        medians = ", ".join(f"{k.split('|')[0][:5]}{k.split('|')[1]}:{g['median_gain']:+.4f}" for k, g in v["groups"].items())
        lines.append(f"      medians: {medians}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("dev", "confirm"))
    parser.add_argument("--directory", type=Path, default=Path("results/stacked_head"))
    parser.add_argument("--chosen", default=None, help="variant fixed by the development stage (required for confirm)")
    args = parser.parse_args()
    if args.stage == "dev":
        rep = report(load_outcomes(args.directory, DEV_TAGS))
        rep["selection"] = select_variant(rep)
        text = format_report(rep, "Development") + f"\nSelection: {rep['selection']['chosen']} ({rep['selection']['reason']})"
        (args.directory / "gate_dev_report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        (args.directory / "gate_dev_report.txt").write_text(text + "\n", encoding="utf-8")
    else:
        if args.chosen is None:
            parser.error("--chosen is required for the confirmation stage")
        rep = report(load_outcomes(args.directory, CONFIRM_TAGS))
        rep["confirmation"] = confirm(rep, args.chosen)
        text = format_report(rep, "Confirmation (fresh sets, test-split evaluation)") + f"\nConfirmation: {json.dumps(rep['confirmation'])}"
        (args.directory / "gate_confirmation_report.json").write_text(json.dumps(rep, indent=1) + "\n", encoding="utf-8")
        (args.directory / "gate_confirmation_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
