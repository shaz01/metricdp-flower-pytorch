"""Merge the EuroSAT CIA IN/OUT result files and compute round-matched AUC.

Reads results/cia_eurosat_scaling/cia_in.json and cia_out.json (written separately by
experiments.cia.scripts.eurosat_scaling --group in|out, so the two runs never race on a shared
report file), pairs each (partition, privacy) combo's IN and OUT trajectories at their shared
checkpoint rounds, and reports round-matched AUC plus a bootstrap 95% interval -- same scoring
primitives as the existing Alzheimer/CIFAR-10/Fashion-MNIST/CIFAR-100 CIA reports
(experiments/cia/reports/build_alzheimer_cia_report.py), not reimplemented. Mirrors
experiments/cia/scripts/cifar100_scaling_analysis.py exactly; the only real differences are the
results directory and the checkpoint-round count (11 here vs. 26 for CIFAR-100's longer sweep,
which changes nothing in this module -- round_matched_auc pairs on whatever rounds are common to
both sides). Pools all 3 seeds (42, 43, 44) per combo -- 33 pairs per combo (3 seeds x 11
checkpoint rounds), 198 pairs total across the 6 combos.

Usage:
    uv run python -m experiments.cia.scripts.eurosat_scaling_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.cia.reports.build_alzheimer_cia_report import (
    attack_scores,
    bootstrap_auc,
    pooled_rows,
    round_matched_auc,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "cia_eurosat_scaling"
BOOTSTRAP_SEED = 42  # RNG seed for the percentile bootstrap resampling -- unrelated to the
# experiment's own training seeds (42, 43, 44), which pooled_rows/round_matched_auc already
# hard-code internally.
SCORE_KEYS = ("target_clean_shadow_loss", "target_noisy_shadow_loss")


def load_results(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def group_by_combo(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    """Group CIA result rows by (mechanism, seed), sorted by server_round.

    mechanism is f"{partition_mode}:{privacy}" -- a formatted string, matching the
    mechanism: str type build_alzheimer_cia_report.py's pooled_rows/round_matched_auc expect
    (a tuple would work at runtime as a dict key but violate that contract).
    """
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        mechanism = f"{row['partition_mode']}:{row['privacy']}"
        key = (mechanism, row["seed"])
        groups.setdefault(key, []).append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: row["server_round"])
    return groups


def _validate_round_alignment(
    in_groups: dict[tuple[str, int], list[dict]],
    out_groups: dict[tuple[str, int], list[dict]],
    mechanism: str,
) -> None:
    """Raise ValueError if any seed's IN/OUT checkpoint rounds don't match exactly.

    round_matched_auc pairs IN/OUT scores positionally per seed via zip(strict=True), which only
    catches a *length* mismatch -- two same-length round lists with different values would be
    silently mispaired without this check. Raises a plain KeyError (uncaught here, handled by the
    caller) if a seed's group is missing entirely for this mechanism on either side.
    """
    for seed in (42, 43, 44):
        in_rounds = [row["server_round"] for row in in_groups[(mechanism, seed)]]
        out_rounds = [row["server_round"] for row in out_groups[(mechanism, seed)]]
        if in_rounds != out_rounds:
            raise ValueError(f"seed {seed} IN/OUT rounds differ: {in_rounds} vs {out_rounds}")


def build_summary(
    in_groups: dict[tuple[str, int], list[dict]],
    out_groups: dict[tuple[str, int], list[dict]],
) -> list[dict]:
    """Build the round-matched AUC summary, tolerating partial/interrupted combos.

    `run_attack` is explicitly built to continue past training failures ("N attempted, M
    failed"), so a single combo (or a single seed within a combo) can end up with a partial or
    mismatched IN/OUT trajectory -- an expected, real outcome that must not abort the whole
    analysis. Any mechanism that can't be scored (missing from one side entirely, missing an
    entire seed's group -- a plain KeyError from pooled_rows/round_matched_auc -- or whose IN/OUT
    rounds don't line up within some seed -- a ValueError from this module's own
    _validate_round_alignment, not from round_matched_auc's own zip(strict=True) check, which only
    catches length mismatches -- is warned about by name and skipped; every other mechanism is
    still reported.
    """
    summary = []
    in_mechanisms = {mechanism for mechanism, _seed in in_groups}
    out_mechanisms = {mechanism for mechanism, _seed in out_groups}
    only_in = sorted(in_mechanisms - out_mechanisms)
    only_out = sorted(out_mechanisms - in_mechanisms)
    for mechanism in only_in:
        print(f"WARNING: mechanism {mechanism!r} has IN rows but no OUT rows -- skipping.")
    for mechanism in only_out:
        print(f"WARNING: mechanism {mechanism!r} has OUT rows but no IN rows -- skipping.")
    for mechanism in sorted(in_mechanisms & out_mechanisms):
        partition_mode, privacy = mechanism.split(":", 1)
        entry: dict = {"partition_mode": partition_mode, "privacy": privacy}
        try:
            _validate_round_alignment(in_groups, out_groups, mechanism)
            for score_key in SCORE_KEYS:
                in_scores = attack_scores(pooled_rows(in_groups, mechanism), score_key)
                out_scores = attack_scores(pooled_rows(out_groups, mechanism), score_key)
                pooled, low, high = bootstrap_auc(in_scores, out_scores, seed=BOOTSTRAP_SEED)
                entry[score_key] = {
                    "round_matched_auc": round_matched_auc(
                        in_groups, out_groups, mechanism, score_key
                    ),
                    "pooled_auc": pooled,
                    "pooled_auc_ci95": [low, high],
                }
        except (ValueError, KeyError) as error:
            print(
                f"WARNING: mechanism {mechanism!r} could not be scored ({error!r}) -- skipping."
            )
            continue
        summary.append(entry)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def _should_refuse_overwrite(summary: list[dict], output_path: Path) -> bool:
    """True if writing `summary` to `output_path` would destroy real existing data.

    `build_summary` returns [] whenever every mechanism failed to score. If `output_path`
    already holds a non-empty result from a prior run, writing an empty summary over it would
    silently discard real committed data. Refuse only that specific case; an empty summary is
    fine to write if there's nothing real to lose (no file yet, or the existing file is already
    empty).
    """
    if summary:
        return False
    if not output_path.exists():
        return False
    try:
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return bool(existing)


def main() -> None:
    args = _parser().parse_args()
    results_dir = args.results_dir.resolve()
    in_groups = group_by_combo(load_results(results_dir / "cia_in.json"))
    out_groups = group_by_combo(load_results(results_dir / "cia_out.json"))
    summary = build_summary(in_groups, out_groups)
    output_path = results_dir / "cia_analysis.json"
    if _should_refuse_overwrite(summary, output_path):
        print(
            f"WARNING: no mechanism could be scored -- refusing to overwrite the existing "
            f"non-empty {output_path}."
        )
        raise SystemExit(1)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(summary)} combo(s) to {output_path}")


if __name__ == "__main__":
    main()
