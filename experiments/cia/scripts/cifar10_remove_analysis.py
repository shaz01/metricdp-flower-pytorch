"""Score the CIFAR-10 removal-adjacency CIA runs the way the paper does.

Reads the per-chunk ``cia.json`` files written by
``experiments.cia.scripts.cifar10_remove`` under
``results/cia/cifar10_remove/clients-<n>/<adjacency>/<privacy>/runs/`` and
reproduces both of Sáinz-Pardo Díaz et al. (2026)'s CIA tables:

* **Tables 10-12 (first-round attack)** -- aggregated test loss, target shadow
  loss, and their relative difference ``(target - aggregated) / target * 100``
  at round 1 of the IN trajectory.
* **Table 13 (multi-round attack)** -- an AUC ranking the IN trajectory's
  per-round scores against the OUT trajectory's, where a round's score is the
  negated mean shadow loss, with a 95% stratified percentile bootstrap
  interval.

Alongside the paper's pooled AUC this also reports the two confound-free
variants this repo already uses for its Alzheimer/CIFAR/Fashion reports: a
round-matched AUC (the pooled 20x20 comparison also ranks an early IN
checkpoint against a late OUT one, mixing membership signal with the round
index) and an aggregate-referenced AUC scored on the relative-difference
column instead of the raw loss.

All scoring primitives are imported from
``experiments/cia/reports/build_alzheimer_cia_report.py`` rather than
reimplemented, so every CIA report in this repo shares one definition of AUC
and of the bootstrap.

Usage:
    uv run python -m experiments.cia.scripts.cifar10_remove_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cia.reports.build_alzheimer_cia_report import (
    SHADOW_KEYS,
    attack_scores,
    auc,
    bootstrap_auc,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "cia" / "cifar10_remove"
DEFAULT_OUTPUT_NAME = "cia_analysis.json"
SEED = 42
PRIVACY_ORDER = ("vanilla", "global-dp", "metric-privacy")


def load_chunk(results_dir: Path, clients: int, adjacency: str, privacy: str) -> list[dict]:
    """Load one chunk's CIA rows, sorted by server round."""
    path = results_dir / f"clients-{clients}" / adjacency / privacy / "runs" / "cia.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows.sort(key=lambda row: int(row["server_round"]))
    return rows


def discover_client_counts(results_dir: Path) -> list[int]:
    """Return every client count with a results directory, ascending."""
    counts = []
    for path in results_dir.glob("clients-*"):
        if path.is_dir():
            try:
                counts.append(int(path.name.split("-", 1)[1]))
            except ValueError:
                continue
    return sorted(counts)


def round_matched_auc(
    in_rows: list[dict], out_rows: list[dict], score_key: str
) -> float:
    """AUC over IN/OUT pairs taken at identical checkpoint rounds.

    Single-seed counterpart of build_alzheimer_cia_report.round_matched_auc,
    which loops over that experiment's three fixed seeds; these runs use one
    seed, matching the sweep they attack.
    """
    in_rounds = [int(row["server_round"]) for row in in_rows]
    out_rounds = [int(row["server_round"]) for row in out_rows]
    if in_rounds != out_rounds:
        raise ValueError("IN and OUT rows must share identical checkpoint rounds.")
    in_scores = attack_scores(in_rows, score_key)
    out_scores = attack_scores(out_rows, score_key)
    outcomes = [
        1.0 if in_score > out_score else 0.5 if in_score == out_score else 0.0
        for in_score, out_score in zip(in_scores, out_scores, strict=True)
    ]
    return float(np.mean(outcomes))


def first_round_entry(in_rows: list[dict]) -> dict:
    """Reproduce the paper's Tables 10-12 row from the IN trajectory's round 1."""
    first = next(row for row in in_rows if int(row["server_round"]) == 1)
    return {
        "aggregated_test_loss": float(first["aggregated_test_loss"]),
        "target_clean_shadow_loss": float(first["target_clean_shadow_loss"]),
        "target_noisy_shadow_loss": float(first["target_noisy_shadow_loss"]),
        "clean_difference_pct": float(first["clean_difference_pct"]),
        "noisy_difference_pct": float(first["noisy_difference_pct"]),
    }


def multi_round_entry(in_rows: list[dict], out_rows: list[dict]) -> dict:
    """Reproduce the paper's Table 13 AUCs for one (clients, privacy) combo."""
    entry: dict = {}
    for shadow_kind, (loss_key, difference_key) in SHADOW_KEYS.items():
        in_scores = attack_scores(in_rows, loss_key)
        out_scores = attack_scores(out_rows, loss_key)
        pooled, low, high = bootstrap_auc(in_scores, out_scores, seed=SEED)
        entry[shadow_kind] = {
            "pooled_auc": pooled,
            "pooled_auc_ci95": [low, high],
            "round_matched_auc": round_matched_auc(in_rows, out_rows, loss_key),
            "aggregate_referenced_auc": auc(
                attack_scores(in_rows, difference_key),
                attack_scores(out_rows, difference_key),
            ),
            "rounds": len(in_rows),
        }
    return entry


def build_summary(results_dir: Path, client_counts: list[int]) -> list[dict]:
    """Score every available (clients, privacy) combo, skipping incomplete ones."""
    summary: list[dict] = []
    for clients in client_counts:
        for privacy in PRIVACY_ORDER:
            try:
                in_rows = load_chunk(results_dir, clients, "in-remove", privacy)
                out_rows = load_chunk(results_dir, clients, "out-remove", privacy)
            except FileNotFoundError as error:
                print(
                    f"WARNING: clients={clients} privacy={privacy!r} is missing a "
                    f"chunk ({error.filename}) -- skipping."
                )
                continue
            if not in_rows or not out_rows:
                print(
                    f"WARNING: clients={clients} privacy={privacy!r} has an empty "
                    "trajectory -- skipping."
                )
                continue
            try:
                entry = {
                    "num_clients_canonical": clients,
                    "num_clients_in": int(in_rows[0]["num_clients"]),
                    "num_clients_out": int(out_rows[0]["num_clients"]),
                    "privacy": privacy,
                    "partition_mode": in_rows[0]["partition_mode"],
                    "noise_multiplier": float(in_rows[0]["noise_multiplier"]),
                    "first_round": first_round_entry(in_rows),
                    "multi_round": multi_round_entry(in_rows, out_rows),
                }
            except (ValueError, StopIteration) as error:
                print(
                    f"WARNING: clients={clients} privacy={privacy!r} could not be "
                    f"scored ({error}) -- skipping."
                )
                continue
            summary.append(entry)
    return summary


def format_tables(summary: list[dict]) -> str:
    """Render the paper's two tables as plain text."""
    lines = [
        "First-round attack (paper Tables 10-12), IN trajectory, round 1",
        f"{'n':>4}  {'privacy':15s} {'agg loss':>9} {'clean loss':>10} "
        f"{'rel diff %':>10} {'noisy loss':>10} {'rel diff %':>10}",
    ]
    for entry in summary:
        first = entry["first_round"]
        lines.append(
            f"{entry['num_clients_canonical']:4d}  {entry['privacy']:15s} "
            f"{first['aggregated_test_loss']:9.3f} "
            f"{first['target_clean_shadow_loss']:10.3f} "
            f"{first['clean_difference_pct']:10.2f} "
            f"{first['target_noisy_shadow_loss']:10.3f} "
            f"{first['noisy_difference_pct']:10.2f}"
        )

    lines += [
        "",
        "Multi-round attack (paper Table 13): AUC (95% bootstrap CI)",
        f"{'n':>4}  {'privacy':15s} {'shadow':6s} {'pooled AUC':>22} "
        f"{'round-matched':>13} {'agg-referenced':>14}",
    ]
    for entry in summary:
        for shadow_kind in ("clean", "noisy"):
            scores = entry["multi_round"][shadow_kind]
            low, high = scores["pooled_auc_ci95"]
            pooled = f"{scores['pooled_auc']:.3f} ({low:.3f}, {high:.3f})"
            lines.append(
                f"{entry['num_clients_canonical']:4d}  {entry['privacy']:15s} "
                f"{shadow_kind:6s} {pooled:>22} "
                f"{scores['round_matched_auc']:13.3f} "
                f"{scores['aggregate_referenced_auc']:14.3f}"
            )
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--clients",
        type=int,
        nargs="+",
        help="Client counts to score (default: every clients-* directory found).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"Summary JSON path (default: <results-dir>/{DEFAULT_OUTPUT_NAME}).",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    results_dir = args.results_dir.resolve()
    client_counts = args.clients or discover_client_counts(results_dir)
    if not client_counts:
        raise SystemExit(f"No clients-* result directories under {results_dir}")
    summary = build_summary(results_dir, client_counts)
    if not summary:
        raise SystemExit("No complete IN/OUT pair could be scored.")
    print(format_tables(summary))
    output_path = args.output or results_dir / DEFAULT_OUTPUT_NAME
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
