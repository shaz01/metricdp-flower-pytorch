"""C1 analysis: the composite (subsampling at q = 0.25 with logit adjustment) against metric privacy at matched accuracy.

Uses A1's matched-accuracy statistic (``results/cia_v2/frontier/a1/analyze.py``): each configuration's AUC is a curve
in mean final accuracy, linear between its levels, and the difference at a target accuracy gets the pair jackknife
(gate) and the two-way bootstrap. Every curve shares M2's design, so each resample draws the same pairs in all of them.

- Gate (G-C1, decision 15): composite against the paper's metric privacy (A1's runs), all candidates.
- Reported: against private median (A1), global DP (M2) and subsampling alone (B1).
- Adaptive attackers (reported): (a) label-matched surrogates, composite against global DP under the same attacker
  (only M2's runs scored surrogates); (b) the defense-aware logit-adjusted loss on the composite, against metric privacy
  under the standard attacker.

    uv run --no-sync python -m results.cia_v2.composite.analyze --composite c1-r...=<ratio> ... \\
        --metric-privacy ... --private-median ... --gdp ... --subsampled ... --out results/cia_v2/composite/analysis.json
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

from results.cia_v2.frontier import analyze as m2
from results.cia_v2.frontier.a1 import analyze as a1
from results.cia_v2.mechanisms.b1_subsampling.analyze import amplified_mu
from results.cia_v2.randomized_inclusion.analysis import load_audit

HERE = Path(__file__).resolve().parent
SAMPLING_RATE = 0.25
REDUCTION = 0.05  # decision 9


def g_c1(matched_all):
    """Pass at a target: AUC at least 0.05 below metric privacy, gate interval below 0 (formal clause reported only)."""
    difference = matched_all["difference"]
    passed = (difference["estimate"] is not None and difference["high"] is not None
              and difference["estimate"] <= -REDUCTION and difference["high"] < 0)
    return {"verdict": "pass" if passed else "fail", "difference": difference,
            "difference_two_way": matched_all["difference_two_way"]}


def with_losses(audits, attribute):
    """The same runs scored by another attacker view (``surrogate_losses`` or ``adjusted_losses``)."""
    out = []
    for audit in audits:
        losses = getattr(audit, attribute)
        if losses is None:
            raise ValueError(f"{audit.config} has no {attribute}")
        out.append(replace(audit, losses=losses, surrogate_losses=None, adjusted_losses=None))
    return out


def report(composite, ratios, references, gdp_surrogate_reference, targets, resamples=m2.RESAMPLES):
    levels = []
    for audit, ratio in zip(composite, ratios):
        summary = m2.config_summary(audit, noise_ratio=ratio, resamples=resamples)
        adjusted = m2.config_summary(with_losses([audit], "adjusted_losses")[0], noise_ratio=ratio, resamples=resamples)
        mu = amplified_mu(ratio, SAMPLING_RATE, rounds=audit.rounds[-1])
        levels.append(a1._json(summary | {"formal_mu": None if math.isinf(mu) else mu, "formal_mu_computable":
                                          math.isfinite(mu), "adjusted_attacker": {k: adjusted[k] for k in
                                                                                    ("all", "top-decile")}}))
    matched = {name: a1.mechanism_report(composite, curve, targets=targets, resamples=resamples)
               for name, curve in references.items()}
    adaptive = {
        "defense_aware_vs_metric_privacy": a1.mechanism_report(
            with_losses(composite, "adjusted_losses"), references["metric_privacy"], targets=targets,
            resamples=resamples),
        "surrogate_vs_global_dp_surrogate": a1.mechanism_report(
            with_losses(composite, "surrogate_losses"), with_losses(gdp_surrogate_reference, "surrogate_losses"),
            targets=targets, resamples=resamples),
    }
    gates = {str(t): g_c1(matched["metric_privacy"]["all"][str(t)]) for t in targets}
    return {"targets": list(targets), "levels": levels, "matched": matched, "adaptive": adaptive, "g_c1": gates}


def main():
    master = HERE.parents[1]  # results/
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "results")
    parser.add_argument("--a1-root", type=Path, default=master / "cia_v2" / "frontier" / "a1" / "results")
    parser.add_argument("--gdp-root", type=Path, default=master / "cia_v2" / "frontier" / "results")
    parser.add_argument("--b1-root", type=Path, default=master / "cia_v2" / "mechanisms" / "b1_subsampling" / "results")
    parser.add_argument("--composite", nargs="+", required=True, help="config=noise_ratio")
    parser.add_argument("--metric-privacy", nargs="+", required=True)
    parser.add_argument("--private-median", nargs="+", required=True)
    parser.add_argument("--gdp", nargs="+", required=True)
    parser.add_argument("--subsampled", nargs="+", required=True)
    parser.add_argument("--targets", nargs="+", type=float, default=list(a1.TARGETS))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    load = lambda root, items: [load_audit(root, config=name) for name, _ in a1._parse(items)]  # noqa: E731
    composite_levels = a1._parse(args.composite)
    gdp = load(args.gdp_root, args.gdp)
    references = {"metric_privacy": load(args.a1_root, args.metric_privacy),
                  "private_median": load(args.a1_root, args.private_median),
                  "global_dp": gdp, "subsampling_q0.25": load(args.b1_root, args.subsampled)}
    result = report(load(args.root, args.composite), [r for _, r in composite_levels], references, gdp,
                    tuple(args.targets))
    text = json.dumps(result, indent=2, allow_nan=False, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
