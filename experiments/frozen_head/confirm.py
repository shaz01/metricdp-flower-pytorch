"""Confirmatory run of B4's scarce-public positive on a fresh layout (owner decision 11; confirm/PROTOCOL.md).

Reads the ``confirm_public`` runs (layout 44, public sizes 20 and 50), refits the public-only baselines on layout 44's
subsets (Amendment 2, part E), recomputes Part A's gains against ``protocol``, ``best_legit`` and ``oracle_c``, and
applies the confirmation rule against ``best_legit`` to the settings that were robust in the original sweep.

    python -m experiments.frozen_head.confirm
"""

from __future__ import annotations

import json

import numpy as np

from experiments.frozen_head import runner
from experiments.frozen_head.baseline_check import best_legitimate, fits, with_baseline

# Robust against best_legit in the original sweep (analysis_baseline_check.json), by public size.
ORIGINAL_ROBUST = {
    20: [(48, 10, 1.0), (48, 10, 2.0), (48, 20, 2.0), (480, 10, 1.0), (480, 10, 2.0), (480, 20, 1.0), (480, 20, 2.0)],
    50: [(48, 10, 2.0), (48, 20, 2.0), (480, 10, 1.0), (480, 10, 2.0), (480, 20, 1.0), (480, 20, 2.0)],
}
OUTPUT = runner.RESULTS.parent / "confirm" / "analysis.json"


def confirmation(reading: dict, original: dict = ORIGINAL_ROBUST) -> dict:
    """Per size: confirmed if every originally robust setting keeps a lower end of at least 5 pp (PROTOCOL Section 4)."""
    lows = {(r["public_size"], r["clients"], r["rounds"], r["mu_total"]): r["gain_ci95_pp"][0]
            for r in reading["settings"]}
    verdicts = {}
    for size, settings in sorted(original.items()):
        held = [s for s in settings if lows[(size, *s)] >= runner.GATE_PP]
        verdict = ("confirmed" if len(held) == len(settings) else "partly confirmed" if held else "not confirmed")
        verdicts[size] = {"verdict": verdict, "held": held, "tested": settings}
    return verdicts


def main() -> None:
    records = runner._load("confirm_public")
    expected = runner.confirm_jobs(runner.select_by_public_size(runner._load("tune_public")))
    if len(records) != len(expected):
        raise RuntimeError(f"{len(records)} of {len(expected)} confirmatory runs present")
    alternatives = fits(runner.CONFIRM_LAYOUT, runner.CONFIRM_SIZES)
    for record in records:
        key = (record["job"]["public_size"], record["job"]["public_subset"])
        if record["public_only"]["server_test_correct"] != alternatives[key]["protocol"]["server_test_correct"]:
            raise RuntimeError(f"{record['name']}: recomputed protocol baseline differs from the recorded one.")
    chosen_by_size = runner.select_by_public_size(runner._load("tune_public"))
    picks = {key: best_legitimate(alt) for key, alt in alternatives.items()}
    readings = {"protocol": runner.analyze_public(records, chosen_by_size)}
    for reading, pick in (("best_legit", picks), ("oracle_c", {key: "oracle_c" for key in alternatives})):
        baselines = {key: alternatives[key][pick[key]]["server_test_correct"] for key in alternatives}
        readings[reading] = runner.analyze_public(with_baseline(records, baselines), chosen_by_size)
    result = {
        "layout_seed": runner.CONFIRM_LAYOUT,
        "confirmation": confirmation(readings["best_legit"]),
        "fits": [{"public_size": size, "subset": subset, "best_legit": picks[(size, subset)],
                  **{name: {"inverse_l2": alt["inverse_l2"], "server_test_acc": float(np.mean(alt["server_test_correct"]))}
                     for name, alt in alternatives[(size, subset)].items()}}
                 for size, subset in sorted(alternatives)],
        "readings": readings,
    }
    runner._atomic_json(OUTPUT, result)
    print(json.dumps({size: v["verdict"] for size, v in result["confirmation"].items()}))


if __name__ == "__main__":
    main()
