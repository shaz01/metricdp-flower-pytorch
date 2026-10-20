"""A1 power from M2's runs: the matched-accuracy difference between two disjoint global-DP curves.

A1's gate interval is the pair jackknife of ``AUC_mechanism(a*) - AUC_GDP(a*)`` over curves that share M2's design.
Splitting M2's six configurations into two curves with no configuration in common (vanilla, 2x, 8x against 1x, 4x,
16x) gives a difference with that structure, and its jackknife standard error. The detectable difference at 80% power
for the two-sided 97.5% t interval is ``(t_0.9875,P-1 + z_0.8) SE``. Both curves interpolate between levels as A1's
will; A1's mechanisms add their own variance, so this is a guide, not a bound.

    uv run --no-sync python -m results.cia_v2.frontier.a1.power
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm, t as student

from experiments.cia.audit_attacks import GATE_LEVEL, _complementary_pairs
from results.cia_v2.frontier.a1.analyze import TARGETS, _Curve, subsets
from results.cia_v2.frontier.a1.analyze import HERE as A1
from results.cia_v2.randomized_inclusion.analysis import load_audit

CURVES = (("vanilla", "gdp-r2x", "gdp-r8x"), ("gdp-r1x", "gdp-r4x", "gdp-r16x"))
POWER = 0.8


def main():
    audits = [[load_audit(A1.parent / "results", config=name) for name in names] for names in CURVES]
    for subset_name, subset in subsets(audits[0][0]).items():
        first, second = _Curve(audits[0], subset), _Curve(audits[1], subset)
        candidates, replicates = first.scores[0].shape
        pairs = _complementary_pairs(first.membership[0])
        rows = np.arange(candidates)[:, None]
        q = student.ppf(0.5 + GATE_LEVEL / 2, len(pairs) - 1) + norm.ppf(POWER)
        for target in TARGETS:
            values = np.array([first.value(target, (rows, np.setdiff1d(np.arange(replicates), p)[None, :]))
                               - second.value(target, (rows, np.setdiff1d(np.arange(replicates), p)[None, :]))
                               for p in pairs])
            n = len(values)
            se = float(np.sqrt((n - 1) / n * np.sum((values - values.mean()) ** 2)))
            estimate = first.value(target) - second.value(target)
            print(f"{subset_name} a*={target}: difference {estimate:+.3f}, jackknife SE {se:.3f}, "
                  f"detectable at 80% power {q * se:.3f}")


if __name__ == "__main__":
    main()
