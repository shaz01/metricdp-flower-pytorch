"""M2 analysis: corrected leakage-accuracy curve of global DP (and vanilla) under the v2 measurement standard.

Reads the randomized-inclusion runs of each configuration (``results/cia_v2/randomized_inclusion/analysis.py``), scores
them with the primary calibrated attacker, and reports per configuration and subset (all candidates, top decile):
AUC with the gate interval (leave-one-pair-out jackknife at ``GATE_LEVEL``) and the two-way bootstrap for reference,
TPR at 5% and 1% FPR, the final-round-only and threshold attackers, accuracy (bootstrap over replicates), the
per-client minimum accuracy, and the formal mu. Then ``r_0.6`` and G-M2 as applied (STATE.md, Gate).

    uv run --no-sync python -m results.cia_v2.frontier.analyze --out results/cia_v2/frontier/analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments.cia.audit_attacks import (
    GATE_LEVEL, calibrated_scores, pair_jackknife, roc_auc, threshold_scores, tpr_at_fpr, two_way_bootstrap,
)
from results.cia_v2.randomized_inclusion.analysis import load_audit
from results.cia_v2.randomized_inclusion.runner import FRONTIER_RATIO

HERE = Path(__file__).resolve().parent
ACTIVE, ROUNDS = 32, 50
RESAMPLES = 2000
BOOTSTRAP_SEED = 2026
FILL_IN = {"2x": (1, 4), "8x": (4, 16)}  # level -> neighbouring measured levels (multiples of the frontier ratio)
FILL_IN_MARGIN = 0.05
LEAKAGE_LIMIT, ACCURACY_FLOOR = 0.6, 0.70


def formal_mu(ratio, active=ACTIVE, rounds=ROUNDS):
    """GDP mu of global DP with equal weights over ``active`` clients: per round 1 / (n r), composed over rounds."""
    return math.inf if ratio <= 0 else math.sqrt(rounds) / (active * ratio)


def fill_in_levels(stage1_accuracy):
    """Fill-in rule (STATE.md, Design): add a level between two measured ones whose mean accuracies differ by > 0.05."""
    by_multiple = {round(r / FRONTIER_RATIO): a for r, a in stage1_accuracy.items()}
    return [level for level, (low, high) in FILL_IN.items()
            if by_multiple[low] - by_multiple[high] > FILL_IN_MARGIN]


def _interval(result):
    return {"estimate": result.estimate, "low": result.low, "high": result.high}


def _subset_metrics(losses, membership, resamples, surrogates=None):
    scores = calibrated_scores(losses, membership)
    return {
        "auc": _interval(pair_jackknife(scores, membership, roc_auc)),
        "auc_two_way": _interval(two_way_bootstrap(scores, membership, roc_auc, resamples=resamples,
                                                   seed=BOOTSTRAP_SEED)),
        "tpr_at_5pct_fpr": tpr_at_fpr(scores, membership, 0.05),
        "tpr_at_1pct_fpr": tpr_at_fpr(scores, membership, 0.01),
        "auc_final_round": roc_auc(calibrated_scores(losses[..., -1:], membership), membership),
        "auc_threshold": roc_auc(threshold_scores(losses), membership),
        # Label-matched public surrogates scored like the shadow samples (B5 diagnostic; no gate).
        "auc_surrogate": (None if surrogates is None
                          else roc_auc(calibrated_scores(surrogates, membership), membership)),
    }


def config_summary(audit, *, noise_ratio, resamples=RESAMPLES):
    top = np.isin(audit.candidates, audit.top_decile)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    accuracy = audit.final_accuracy
    draws = [accuracy[rng.integers(len(accuracy), size=len(accuracy))].mean() for _ in range(resamples)]
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "config": audit.config, "noise_ratio": noise_ratio, "noise_multiple": noise_ratio / FRONTIER_RATIO,
        # vanilla: no guarantee; rounds = the last checkpoint (EuroSAT 50, Alzheimer 100)
        "formal_mu": formal_mu(noise_ratio, rounds=audit.rounds[-1]) if noise_ratio > 0 else None,
        "replicates": int(audit.membership.shape[1]),
        "gate_level": GATE_LEVEL,
        "all": _subset_metrics(audit.losses, audit.membership, resamples, audit.surrogate_losses),
        "top-decile": _subset_metrics(audit.losses[top], audit.membership[top], resamples,
                                      None if audit.surrogate_losses is None else audit.surrogate_losses[top]),
        "accuracy": {"estimate": float(accuracy.mean()), "low": float(low), "high": float(high)},
        "min_client_accuracy": (None if audit.min_client_accuracy is None
                                else float(np.mean(audit.min_client_accuracy))),
    }


def g_m2(summaries):
    """``r_0.6``: the lowest DP noise meeting all three conditions; operating points ``r_0.6`` and the next lower."""
    dp = sorted((s for s in summaries if s["noise_ratio"] > 0), key=lambda s: s["noise_ratio"])
    useful = [s for s in dp if s["all"]["auc"]["high"] <= LEAKAGE_LIMIT
              and s["top-decile"]["auc"]["estimate"] <= LEAKAGE_LIMIT
              and s["accuracy"]["low"] >= ACCURACY_FLOOR]
    if not useful:
        return {"verdict": "no useful regime", "r_0.6": None, "operating_points": []}
    r = useful[0]["noise_ratio"]
    lower = [s["noise_ratio"] for s in dp if s["noise_ratio"] < r]
    return {"verdict": "useful empirical regime", "r_0.6": r, "operating_points": [r] + lower[-1:],
            "accuracy_at_r_0.6": useful[0]["accuracy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=HERE / "results")
    parser.add_argument("--configs", nargs="+", required=True, help="config=noise_ratio, e.g. gdp-r1x=0.001546")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    summaries = []
    for item in args.configs:
        name, ratio = item.split("=")
        summaries.append(config_summary(load_audit(args.root, config=name), noise_ratio=float(ratio)))
    report = {"configs": summaries, "g_m2": g_m2(summaries)}
    text = json.dumps(report, indent=2, allow_nan=False, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
