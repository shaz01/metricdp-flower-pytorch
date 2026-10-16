"""Step 1 analysis: WP2's statistics and G2 rule over seeds 42-45, with consensus-kl-0.3 at seeds 43-45 added.

Loads WP2's committed trajectories (seeds 42-45 for vanilla and the controls, seed 42 for every configuration) and
this follow-up's consensus-kl-0.3 trajectories (seeds 43-45), merges them per configuration, keeps the configurations
that cover all four seeds and carries the other WP2 candidates' seed-42 failures (WP2 analyze.py, Amendment 1
mechanics). Nothing in WP2's code changes.

    uv run --no-sync python -m results.cia_frontier.consensus_kl.analyze --out results/cia_frontier/consensus_kl/analysis_step1.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from results.cia_frontier.signature_suppression import analyze as wp2

ROOT = Path(__file__).resolve().parents[3]
WP2_RESULTS = ROOT / "results/cia_frontier/signature_suppression/results/stage_a"
WP2_SEED42 = ROOT / "results/cia_frontier/signature_suppression/analysis.json"
STEP1_RESULTS = ROOT / "results/cia_frontier/consensus_kl/results/seeds43-45"


def merged():
    data = wp2.load(WP2_RESULTS)
    for config, extra in wp2.load(STEP1_RESULTS).items():
        base = data[config]
        if set(base["cells"]) & set(extra["cells"]):
            raise ValueError(f"{config}: overlapping target-seed pairs")
        base["cells"].update(extra["cells"])
        for key in ("accuracy", "in_accuracy", "client_accuracy", "loo_median"):
            base[key] = base[key] + extra[key]
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--resamples", type=int, default=wp2.RESAMPLES)
    args = parser.parse_args()
    data = wp2.extension_subset(merged())
    carried = json.loads(WP2_SEED42.read_text())["gate"]["candidates"]
    text = json.dumps(wp2.evaluate(data, args.resamples, carried), indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
