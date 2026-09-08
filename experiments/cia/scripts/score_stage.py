"""Dataset-agnostic scoring for one AUC-targeted-sweep stage.

Given one stage's ``CiaResult`` rows (both adjacencies, one seed, one privacy
mode -- exactly what a per-dataset ``run_stage()`` returns) and the directory
those runs were written to, compute the round-matched clean-shadow attack AUC
(with direction-reversal) and the averaged final-round accuracy. Ports the
scoring logic already established in ``reports/build_cia_takeaways.py``
(``round_matched_auc_from_rows``/``late_accuracy``) into a single reusable,
dataset-agnostic place instead of that logic living per-dataset.
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from experiments.cia.result import CiaResult


@dataclass(frozen=True)
class StageScore:
    auc: float
    accuracy: float


def effective_auc(value: float) -> float:
    """AUC after letting an attacker choose the more revealing score direction."""
    return max(value, 1.0 - value)


def round_matched_clean_shadow_auc(results: Sequence[CiaResult]) -> float:
    """Round-matched (same-round, IN-vs-OUT) clean-shadow AUC for one stage.

    Expects exactly one "in-remove" and one "out-remove" run name among
    ``results`` (i.e. one stage: one seed, one privacy mode, both adjacencies).
    """
    trajectories: dict[str, list[CiaResult]] = {}
    for result in results:
        trajectories.setdefault(result.run_name, []).append(result)

    in_name = next(name for name in trajectories if "in-remove" in name)
    out_name = next(name for name in trajectories if "out-remove" in name)

    def _scores(name: str) -> list[float]:
        rows = sorted(trajectories[name], key=lambda row: row.server_round)
        return [-row.target_clean_shadow_loss for row in rows]

    in_scores = _scores(in_name)
    out_scores = _scores(out_name)
    directional_auc = statistics.fmean(
        1.0 if score_in > score_out else 0.5 if score_in == score_out else 0.0
        for score_in, score_out in zip(in_scores, out_scores, strict=True)
    )
    return effective_auc(directional_auc)


def stage_accuracy(output_dir: Path, run_names: Sequence[str]) -> float:
    """Average each run's final-round accuracy, reading its own result JSON."""
    values: list[float] = []
    for run_name in run_names:
        payload = json.loads((output_dir / f"{run_name}.json").read_text())
        metrics = payload["server_evaluate_metrics"]
        final_round = max(metrics, key=lambda round_key: int(round_key))
        values.append(float(metrics[final_round]["accuracy"]))
    return statistics.fmean(values)


def score_stage(results: Sequence[CiaResult], output_dir: Path) -> StageScore:
    """Compute both the attack AUC and the accuracy for one stage."""
    run_names = sorted({result.run_name for result in results})
    return StageScore(
        auc=round_matched_clean_shadow_auc(results),
        accuracy=stage_accuracy(output_dir, run_names),
    )
