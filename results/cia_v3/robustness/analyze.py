"""P3 analysis: the every-round sampling-aware attacker and the IN-count interference check (PROTOCOL.md Section 2).

The reproduction check runs first: C1's composite minus metric privacy at 0.740 must give the published
-0.095 [-0.137, -0.053] to 3 decimals, or nothing new is reported.

    uv run --no-sync python -m results.cia_v3.robustness.analyze --out results/cia_v3/robustness/analysis.json
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from experiments.cia.audit_attacks import (
    GATE_LEVEL, _jackknife_interval, calibrated_scores, pair_jackknife, roc_auc, tpr_at_fpr,
)
from research.cia_roadmap_v3.reanalysis import reanalyze as rz
from results.cia_v2.frontier import analyze as m2
from results.cia_v2.frontier.a1 import analyze as a1
from results.cia_v2.randomized_inclusion.analysis import load_audit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TARGETS = (0.740, 0.660)
COMPOSITE = {"p3-comp-r0.000773": 0.000773, "p3-comp-r0.003092": 0.003092}
METRIC_PRIVACY = {"p3-mp-r1x": 0.001546, "p3-mp-r2x": 0.003092}
IN_COUNT = {8: "p3-gdp-r2x-in8", 24: "p3-gdp-r2x-in24"}
BLOCK = 4  # replicates per block of the 8-IN design and of its complement
CHECKPOINT_EVERY = 5  # the standard attacker's 11 checkpoints
REDUCTION = 0.05  # G-C1
INTERFERENCE_LIMIT = 0.03  # G-P3(b)
NOT_TESTABLE = "not testable"


def reproduce_c1(c1_root=REPO / "results/cia_v2/composite/results", a1_root=REPO / "results/cia_v2/frontier/a1/results"):
    """C1's composite minus metric privacy at 0.740 from the committed runs."""
    composite = [rz.standard(rz.load_config(c1_root, name)) for name in rz.C1]
    mp = [rz.standard(rz.load_config(a1_root, name)) for name in rz.MP]
    gate = rz.matched(composite, mp, 0.740, resamples=1)["gate"]
    return gate["estimate"], gate["low"], gate["high"]


def standard_checkpoints(audit):
    """The audit at rounds 0, 5, ..., 50: the gated attackers' view of a run scored at every round."""
    keep = [i for i, t in enumerate(audit.rounds) if t % CHECKPOINT_EVERY == 0]
    return replace(audit, rounds=tuple(audit.rounds[i] for i in keep), losses=audit.losses[..., keep],
                   adjusted_losses=None if audit.adjusted_losses is None else audit.adjusted_losses[..., keep])


def every_round_statistic(losses, schedules, rate):
    """``z = sum_t (s_t - q) (d_t - dbar_{c,t})``, (candidates, replicates): ``rz.schedule_statistic`` with one-round
    windows. ``d_t = x_{t-1} - x_t`` and ``dbar_{c,t}`` is the candidate's mean drop over all its replicates, so no
    membership label is used and the common loss decay is removed."""
    drops = losses[..., :-1] - losses[..., 1:]
    drops = drops - drops.mean(axis=1, keepdims=True)
    return ((schedules.astype(np.float64) - rate) * drops).sum(axis=2)


def every_round_scored(config, *, all_checkpoints=False, z_only=False):
    """The composite under the every-round attacker; ``config`` holds the audit with all 51 checkpoints."""
    audit = config.audit
    z = every_round_statistic(audit.losses, rz.simulated_schedules(config), config.sampling_rate)
    z = z / z.std()  # one constant per configuration, independent of membership
    losses = audit.losses if all_checkpoints else standard_checkpoints(audit).losses
    features = z[..., None] if z_only else np.concatenate([losses, z[..., None]], axis=2)
    return rz.Scored(calibrated_scores(features, audit.membership), audit.membership, audit.final_accuracy)


def scored(audit, losses=None):
    losses = audit.losses if losses is None else losses
    return rz.Scored(calibrated_scores(losses, audit.membership), audit.membership, audit.final_accuracy)


def blocks_of(replicates, block=BLOCK):
    if replicates % block:
        raise ValueError("replicates must be a whole number of blocks")
    return [np.arange(start, start + block) for start in range(0, replicates, block)]


def block_jackknife(metric_of_kept, replicates, *, block=BLOCK, level=GATE_LEVEL):
    """Leave-one-block-out jackknife of ``metric_of_kept(kept replicates)``, t quantiles with blocks - 1 df."""
    everything = np.arange(replicates)
    values = np.array([metric_of_kept(np.setdiff1d(everything, b)) for b in blocks_of(replicates, block)])
    return _jackknife_interval(metric_of_kept(everything), values, level)


def _auc(scores, membership, kept):
    return roc_auc(scores[:, kept].ravel(), membership[:, kept].ravel())


def in_count_difference(eight, twenty_four):
    """``AUC(8 IN) - AUC(24 IN)`` with the block jackknife; both arguments are ``Scored``."""
    return block_jackknife(lambda kept: _auc(eight.scores, eight.membership, kept)
                           - _auc(twenty_four.scores, twenty_four.membership, kept), eight.scores.shape[1])


def _interval(result):
    return {"estimate": result.estimate, "low": result.low, "high": result.high}


def in_count_summary(audit):
    """What ``m2.config_summary`` reports, with the block jackknife: complementary pairs do not exist at 8 or 24 IN."""
    s = scored(audit)
    top = np.isin(audit.candidates, audit.top_decile)
    replicates = audit.membership.shape[1]
    return {"config": audit.config, "active": 16 + int(audit.membership[:, 0].sum()),
            "all": {"auc": _interval(block_jackknife(lambda k: _auc(s.scores, s.membership, k), replicates)),
                    "tpr_at_5pct_fpr": tpr_at_fpr(s.scores, s.membership, 0.05),
                    "tpr_at_1pct_fpr": tpr_at_fpr(s.scores, s.membership, 0.01)},
            "top-decile": {"auc": _interval(block_jackknife(
                lambda k: _auc(s.scores[top], s.membership[top], k), replicates))},
            "accuracy": float(audit.final_accuracy.mean()),
            "min_client_accuracy": float(np.mean(audit.min_client_accuracy))}


def _missing(value):
    return value is None or math.isnan(value)


def c1_outcome(gate):
    """G-C1's rule at a target: at least 0.05 below metric privacy with the gate interval below 0, or not testable."""
    if any(_missing(gate[key]) for key in ("estimate", "low", "high")):
        return NOT_TESTABLE
    return "pass" if gate["estimate"] <= -REDUCTION and gate["high"] < 0 else "fail"


def g_p3a(outcomes):
    """(a) over the targets: fail at any testable target, not testable if none is, otherwise pass."""
    if "fail" in outcomes:
        return "fail"
    return NOT_TESTABLE if all(o == NOT_TESTABLE for o in outcomes) else "pass"


def g_p3b(interval):
    """(b): the whole interval inside +-0.03 (STATE.md, Gate), otherwise escalate."""
    inside = -INTERFERENCE_LIMIT <= interval["low"] and interval["high"] <= INTERFERENCE_LIMIT
    return "pass" if inside else "escalate: IN-count correction, G-P2b provisional"


def lowest_client_at(audits, target):
    points = sorted((float(a.final_accuracy.mean()), float(np.mean(a.min_client_accuracy))) for a in audits)
    accuracy, lowest = np.array(points).T
    return float(np.interp(target, accuracy, lowest)) if accuracy[0] <= target <= accuracy[-1] else None


def analyze(root, resamples=m2.RESAMPLES):
    every_round = [rz.load_config(root, name) for name in COMPOSITE]
    composite = [replace(c, audit=standard_checkpoints(c.audit)) for c in every_round]
    mp = [standard_checkpoints(load_audit(root, config=name)) for name in METRIC_PRIVACY]
    in_count = {k: load_audit(root, config=name) for k, name in IN_COUNT.items()}
    reference = [scored(a) for a in mp]

    attackers = {"every_round": [every_round_scored(c) for c in every_round],
                 "sampling_aware_11": [rz.sampling_aware(c) for c in composite],
                 "standard_11": [scored(c.audit) for c in composite],
                 "all_51": [scored(c.audit) for c in every_round],
                 "all_51_plus_z": [every_round_scored(c, all_checkpoints=True) for c in every_round],
                 "z_only": [every_round_scored(c, z_only=True) for c in every_round]}
    matched = {name: {str(t): rz.matched(points, reference, t, resamples=resamples) for t in TARGETS}
               for name, points in attackers.items() if name != "z_only"}
    per_level = {}
    for i, (name, ratio) in enumerate(COMPOSITE.items()):
        per_level[name] = m2.config_summary(composite[i].audit, noise_ratio=ratio, resamples=resamples) | {
            "realized_sampling_rate": rz.realized_rate(composite[i]),
            "attackers": {key: _interval(pair_jackknife(points[i].scores, points[i].membership, roc_auc))
                          for key, points in attackers.items()}}
    for audit, ratio in zip(mp, METRIC_PRIVACY.values()):
        per_level[audit.config] = m2.config_summary(audit, noise_ratio=ratio, resamples=resamples)

    eight, twenty_four = scored(in_count[8]), scored(in_count[24])
    difference = _interval(in_count_difference(eight, twenty_four))
    outcomes = [c1_outcome(matched["every_round"][str(t)]["gate"]) for t in TARGETS]
    audits = [c.audit for c in composite]
    return a1._json({
        "targets": list(TARGETS), "levels": per_level, "matched": matched,
        "in_count": {"8": in_count_summary(in_count[8]), "24": in_count_summary(in_count[24]),
                     "difference": difference},
        "lowest_client_accuracy_at_0.740": {"composite": lowest_client_at(audits, 0.740),
                                            "metric-privacy": lowest_client_at(mp, 0.740)},
        "decisions": {"g_p3a_by_target": dict(zip(map(str, TARGETS), outcomes)), "g_p3a": g_p3a(outcomes),
                      "g_p3b": g_p3b(difference)}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "results")
    parser.add_argument("--reproduce", action="store_true", help="run only the reproduction check")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    reproduced = reproduce_c1()
    print(json.dumps({"reproduction_c1_vs_mp_at_0.740": reproduced}))
    if tuple(round(x, 3) for x in reproduced) != rz.PUBLISHED_C1_VS_MP[0.740]:
        raise SystemExit(f"reproduction failed: {reproduced} != {rz.PUBLISHED_C1_VS_MP[0.740]}")
    if args.reproduce:
        return
    out = {"reproduction_c1_vs_mp_at_0.740": list(reproduced)} | analyze(args.root)
    text = json.dumps(out, indent=1, allow_nan=False, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
