"""B5 analysis: each intervention against M2's global-DP runs at the same noise (STATE.md, Attacker, metrics, gate).

Every B5 configuration shares M2's design (``design_seed = 2``), so replicate ``j`` of an intervention and of M2's
configuration at the same noise form a pair. Per subset (all candidates, top decile) the paired difference
``AUC_B5 - AUC_GDP`` gets the gate interval (pair jackknife at ``GATE_LEVEL``, the same pairs left out of both) and
the two-way difference bootstrap; accuracy gets a paired bootstrap over replicates.

    uv run --no-sync python -m results.cia_v2.mechanisms.b5_large_clients.analyze \\
        --pairs b5-cap326-r2x=gdp-r2x:0.003092 ... --out results/cia_v2/mechanisms/b5_large_clients/analysis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cia.audit_attacks import (
    GATE_LEVEL, calibrated_scores, difference_bootstrap, pair_jackknife_difference, roc_auc,
)
from results.cia_v2.frontier import analyze as m2
from results.cia_v2.randomized_inclusion.analysis import load_audit
from results.cia_v2.randomized_inclusion.runner import CANDIDATES, TOP_DECILE

HERE = Path(__file__).resolve().parent
REDUCTION, ACCURACY_LOSS = 0.05, 0.01  # G-B5 as amended by owner decision 9
SUBSETS = {"all": np.ones(len(CANDIDATES), dtype=bool), "top-decile": np.isin(CANDIDATES, TOP_DECILE)}


def _interval(result):
    return {"estimate": result.estimate, "low": result.low, "high": result.high}


def paired_difference(intervention, gdp, *, resamples=m2.RESAMPLES, seed=m2.BOOTSTRAP_SEED):
    """AUC and accuracy differences of ``intervention`` minus ``gdp`` (two audits of one design)."""
    if not np.array_equal(intervention.membership, gdp.membership):
        raise ValueError("the intervention and the global-DP configuration must share the inclusion design")
    result = {}
    for name, subset in SUBSETS.items():
        a = calibrated_scores(intervention.losses[subset], intervention.membership[subset])
        b = calibrated_scores(gdp.losses[subset], gdp.membership[subset])
        member = intervention.membership[subset]
        result[name] = {
            "difference": _interval(pair_jackknife_difference(a, member, b, member, roc_auc)),
            "difference_two_way": _interval(difference_bootstrap(a, member, b, member, roc_auc, resamples=resamples,
                                                                 seed=seed)),
        }
    rng = np.random.default_rng(seed)
    delta = intervention.final_accuracy - gdp.final_accuracy
    draws = [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(resamples)]
    low, high = np.percentile(draws, [2.5, 97.5])
    result["accuracy_difference"] = {"estimate": float(delta.mean()), "low": float(low), "high": float(high)}
    return result


def g_b5(difference):
    """Pass: all-candidate AUC at least 0.05 lower, gate interval below 0, accuracy loss at most 1 pp."""
    auc = difference["all"]["difference"]
    accuracy = difference["accuracy_difference"]["estimate"]
    passed = auc["estimate"] <= -REDUCTION and auc["high"] < 0 and accuracy >= -ACCURACY_LOSS
    return {"verdict": "pass" if passed else "fail", "auc_reduction": -auc["estimate"], "accuracy_loss": -accuracy,
            "gate_level": GATE_LEVEL}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "results")
    parser.add_argument("--gdp-root", type=Path, default=HERE.parents[1] / "frontier" / "results")
    parser.add_argument("--pairs", nargs="+", required=True, help="intervention=gdp_config:noise_ratio")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {}
    for item in args.pairs:
        name, rest = item.split("=")
        gdp_name, ratio = rest.split(":")
        intervention, gdp = load_audit(args.root, config=name), load_audit(args.gdp_root, config=gdp_name)
        difference = paired_difference(intervention, gdp)
        report[name] = {"against": gdp_name, "noise_ratio": float(ratio),
                        "intervention": m2.config_summary(intervention, noise_ratio=float(ratio)),
                        "global_dp": m2.config_summary(gdp, noise_ratio=float(ratio)),
                        "paired": difference, "g_b5": g_b5(difference)}
    text = json.dumps(report, indent=2, allow_nan=False, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
