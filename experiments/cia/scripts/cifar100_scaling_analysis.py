"""Merge the CIFAR-100 CIA IN/OUT result files and compute round-matched AUC.

Reads results/cia_cifar100_scaling/cia_in.json and cia_out.json (written separately by
experiments.cia.scripts.cifar100_scaling --group in|out, so the two runs never race on a
shared report file), pairs each (partition, privacy) combo's IN and OUT trajectories at their
shared checkpoint rounds, and reports round-matched AUC plus a bootstrap 95% interval -- same
scoring primitives as the existing Alzheimer/CIFAR-10/Fashion-MNIST CIA reports
(experiments/cia/reports/build_alzheimer_cia_report.py), not reimplemented.

Usage:
    uv run python -m experiments.cia.scripts.cifar100_scaling_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cia.reports.build_alzheimer_cia_report import (
    attack_scores,
    bootstrap_auc,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "cia_cifar100_scaling"
SEED = 42
SCORE_KEYS = ("target_clean_shadow_loss", "target_noisy_shadow_loss")


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_by_combo(rows: list[dict]) -> dict[tuple[str, str], list[dict]]:
    """Group CIA result rows by (partition_mode, privacy), sorted by server_round."""
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["partition_mode"], row["privacy"])
        groups.setdefault(key, []).append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: row["server_round"])
    return groups


def round_matched_auc(
    in_rows: list[dict], out_rows: list[dict], score_key: str
) -> float:
    """Round-matched AUC: pair IN/OUT scores at identical checkpoint rounds.

    Single-seed equivalent of build_alzheimer_cia_report.round_matched_auc, which hard-loops
    over that report's three fixed seeds -- this experiment uses one seed (matching the sweep).
    """
    in_rounds = [row["server_round"] for row in in_rows]
    out_rounds = [row["server_round"] for row in out_rows]
    if in_rounds != out_rounds:
        raise ValueError("IN and OUT rows must share identical checkpoint rounds.")
    in_scores = attack_scores(in_rows, score_key)
    out_scores = attack_scores(out_rows, score_key)
    outcomes = [
        1.0 if in_score > out_score else 0.5 if in_score == out_score else 0.0
        for in_score, out_score in zip(in_scores, out_scores, strict=True)
    ]
    return float(np.mean(outcomes))


def build_summary(
    in_groups: dict[tuple[str, str], list[dict]],
    out_groups: dict[tuple[str, str], list[dict]],
) -> list[dict]:
    """Build the round-matched AUC summary, tolerating partial/interrupted combos.

    `run_attack` is explicitly built to continue past training failures ("N attempted, M
    failed"), so a single combo with a partial or mismatched IN/OUT trajectory is an expected,
    real outcome -- it must not abort the whole analysis. Any combo that can't be scored (missing
    from one side entirely, or whose IN/OUT rounds don't line up) is warned about by name and
    skipped; every other combo is still reported.
    """
    summary = []
    only_in = sorted(set(in_groups) - set(out_groups))
    only_out = sorted(set(out_groups) - set(in_groups))
    for partition, privacy in only_in:
        print(
            f"WARNING: combo partition={partition!r} privacy={privacy!r} has IN rows but no "
            "OUT rows -- skipping."
        )
    for partition, privacy in only_out:
        print(
            f"WARNING: combo partition={partition!r} privacy={privacy!r} has OUT rows but no "
            "IN rows -- skipping."
        )
    for combo_key in sorted(set(in_groups) & set(out_groups)):
        partition, privacy = combo_key
        in_rows = in_groups[combo_key]
        out_rows = out_groups[combo_key]
        entry: dict = {"partition_mode": partition, "privacy": privacy}
        try:
            for score_key in SCORE_KEYS:
                in_scores = attack_scores(in_rows, score_key)
                out_scores = attack_scores(out_rows, score_key)
                pooled, low, high = bootstrap_auc(in_scores, out_scores, seed=SEED)
                entry[score_key] = {
                    "round_matched_auc": round_matched_auc(in_rows, out_rows, score_key),
                    "pooled_auc": pooled,
                    "pooled_auc_ci95": [low, high],
                }
        except ValueError as error:
            print(
                f"WARNING: combo partition={partition!r} privacy={privacy!r} could not be "
                f"scored ({error}) -- skipping."
            )
            continue
        summary.append(entry)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def main() -> None:
    args = _parser().parse_args()
    results_dir = args.results_dir.resolve()
    in_groups = group_by_combo(load_results(results_dir / "cia_in.json"))
    out_groups = group_by_combo(load_results(results_dir / "cia_out.json"))
    summary = build_summary(in_groups, out_groups)
    output_path = results_dir / "cia_analysis.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary)} combo(s) to {output_path}")


if __name__ == "__main__":
    main()
