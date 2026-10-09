"""Deterministic analytical risk calculations; no training or CIA measurements.

Run: uv run python research/calculations/heterogeneous_profile_risk.py
Output is a bounded grid search, not a certified continuous global optimum.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

BUDGETS = (1.0, 2.0, 4.0, 8.0, 16.0)
CURVATURES = (0.0, 0.01, 0.1, 0.5, 1.0)
LAWS = {"ellipsoid": 12.0, "diamond": 8.0}


def directional(total: float, xi: float, h: float, c: float, s: float = 0.8):
    eta = total - xi
    q = 1.0 / (1.0 + math.exp(xi))
    p = 1.0 - q
    t = c / eta**2
    noise_long, noise_short = p + q * h, p * h + q
    long = p * s / (p + t * noise_long)
    short = q * s / (q + t * noise_short)
    risk = (p * (s - long)**2 + q * (s - short)**2
            + t * (noise_long * long**2 + noise_short * short**2))
    return risk, long, short


def amplitude(total: float, xi: float, c: float):
    # Equiprobable scalar magnitudes 0 and 1, isotropic loss and two circle/diamond profiles.
    eta = total - xi
    q = 1.0 / (1.0 + math.exp(xi))
    p = 1.0 - q
    t = 2.0 * c / eta**2
    large = p / (p + t)
    small = q / (q + t)
    risk = 0.5 * (p * (1.0 - large)**2 + q * (1.0 - small)**2
                  + t * (large**2 + small**2))
    return risk, large, small


def search(fn, total, *args, intervals=4096):
    # xi=0 is allowed: the two optimal profiles coincide, recovering the public control.
    best = (*fn(total, 0.0, *args), 0.0)
    for j in range(1, intervals):
        xi = total * j / intervals
        trial = (*fn(total, xi, *args), xi)
        if trial[0] < best[0]:
            best = trial
    return {"risk": best[0], "long_or_large": best[1],
            "short_or_small": best[2], "selector_epsilon": best[3]}


def validate():
    # Independent closed-form public boundary and vanishing-selector cases.
    for total in BUDGETS:
        for c in LAWS.values():
            for h in CURVATURES:
                expected = 0.64 * c * (1.0 + h) / (total**2 + c * (1.0 + h))
                assert math.isclose(directional(total, 0.0, h, c)[0], expected, rel_tol=1e-12)
            expected = 0.5 * 4.0 * c / (total**2 + 4.0 * c)
            assert math.isclose(amplitude(total, 0.0, c)[0], expected, rel_tol=1e-12)
            # h=0 must reduce to isotropic radii and worsen when spending selector budget.
            risk, long, short = directional(total, total / 3.0, 0.0, c)
            assert math.isclose(long, short, rel_tol=1e-12)
            assert risk > directional(total, 0.0, 0.0, c)[0]
    # Independent centered finite-difference check of optimized semiaxis stationarity.
    total, xi, h, c = 8.0, 1.5, 0.1, 12.0
    _, long, short = directional(total, xi, h, c)
    q = 1.0 / (1.0 + math.exp(xi)); p = 1.0 - q
    t = c / (total - xi)**2
    def objective(a, b):
        return p * (0.8-a)**2 + q * (0.8-b)**2 + t*((p+q*h)*a*a+(p*h+q)*b*b)
    step = 1e-6
    assert abs((objective(long+step,short)-objective(long-step,short))/(2*step)) < 1e-8
    assert abs((objective(long,short+step)-objective(long,short-step))/(2*step)) < 1e-8


def main():
    validate()
    rows = []
    for family in ("directional_curvature", "amplitude_zero_one"):
        for total in BUDGETS:
            for h in (CURVATURES if family == "directional_curvature" else (None,)):
                laws = {}
                for law, c in LAWS.items():
                    fn = directional if h is not None else amplitude
                    args = (h,c) if h is not None else (c,)
                    coarse = search(fn,total,*args,intervals=4096)
                    fine = search(fn,total,*args,intervals=8192)
                    laws[law] = {"public_risk": fn(total,0.0,*args)[0],
                                 "private_grid_best": fine,
                                 "grid_risk_change": abs(coarse["risk"]-fine["risk"])}
                public = min(item["public_risk"] for item in laws.values())
                private = min(item["private_grid_best"]["risk"] for item in laws.values())
                projected_public = 8.0 / (total**2 + 16.0) if h is None else None
                if projected_public is not None:
                    public = min(public, projected_public)
                rows.append({"population": family,"epsilon_total":total,"h":h,"laws":laws,
                             "best_public_risk":public,"best_private_grid_risk":private,
                             "public_scalar_projection_risk":projected_public,
                             "relative_gain_percent":100*(public-private)/public})
    result = {"kind":"own deterministic analytical calculation, not FL/CIA evidence",
              "dimension":2,"privacy":"one-release whole-input pure DP reference",
              "selector":"binary RR; xi=0 degenerate public boundary included",
              "search":"uniform xi grids 4096 and 8192 intervals; geometry optimized in closed form",
              "limitations":["not continuous global-optimality certificate",
                             "population-known tuning, not a private estimator",
                             "client-weighted local proxy is not global FL utility",
                             "amplitude profile banks optimize isotropic radii only; scalar projection is an extra public control",
                             "public laws beyond listed bodies/projection not optimized"],
              "checks":"public boundary, h=0 identity, numerical geometry stationarity passed",
              "rows":rows}
    output = Path("results/client_specific_noise/analytical_heterogeneity.json")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(f"Wrote {len(rows)} analytical cases to {output}")
    winners = [r for r in rows if r["relative_gain_percent"] > 1e-8]
    print("Strict grid improvements:",json.dumps([{k:r[k] for k in
          ("population","epsilon_total","h","relative_gain_percent")} for r in winners]))
    print("Maximum grid refinement change:",max(v["grid_risk_change"] for r in rows for v in r["laws"].values()))


if __name__ == "__main__":
    main()
