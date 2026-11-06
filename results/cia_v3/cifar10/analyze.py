"""P2 analysis: CIFAR-10 replication of A-3 and D-4 (PROTOCOL.md Sections 2 and 5).

The reproduction check runs first: with A1's committed EuroSAT runs, metric privacy against global DP at 0.740 must
give A1's published -0.000 [-0.019, 0.018] to 3 decimals, or nothing new is reported.

Stage a (G-P2a): M2's ``config_summary`` (calibrated attacker, 11 checkpoints, pair jackknife at ``GATE_LEVEL``,
two-way bootstrap, top decile, accuracy) on vanilla's replicates 0-7, membership checked against the first 8 columns
of the ``design_seed = 2`` design. Pass if the all-candidate point estimate is at least 0.65.

Pilot (PROTOCOL.md Section 3, accuracy only): mean final server-test accuracy of the two pilot runs per level; targets
``t2``, ``t4`` from global DP at 2x and 4x, the safeguard, and the three-level window for metric privacy and the
composite.

    uv run --no-sync python -m results.cia_v3.cifar10.analyze --stage a --out results/cia_v3/cifar10/analysis_stage_a.json
    uv run --no-sync python -m results.cia_v3.cifar10.analyze --stage pilot --out results/cia_v3/cifar10/pilot_levels.json

Stage b (PROTOCOL.md Section 5, G-P2b): per-configuration summaries; targets ``a2``, ``a4`` from global DP's 32 replicates;
A1's matched-accuracy difference for G-A1 (metric privacy against global DP) and G-C1 (composite against metric privacy,
standard and sampling-aware attackers). The composite is scored at every round; the gated attackers use its 11 standard
checkpoints, and the 51-checkpoint attacker is reported only.

    uv run --no-sync python -m results.cia_v3.cifar10.analyze --stage b --out results/cia_v3/cifar10/analysis.json
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path

import numpy as np

from experiments.cia.audit_attacks import calibrated_scores, pair_jackknife, roc_auc
from research.cia_roadmap_v3.reanalysis import reanalyze as rz
from results.cia_v2.frontier import analyze as m2
from results.cia_v2.frontier.a1 import analyze as a1
from results.cia_v2.randomized_inclusion.analysis import Audit, load_audit
from results.cia_v2.randomized_inclusion.runner import design_matrix

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STAGE_A_REPLICATES = 8
G_P2A_THRESHOLD = 0.65
A1_PUBLISHED = (-0.000, -0.019, 0.018)  # metric privacy minus global DP at 0.740, all candidates (A1 FINDINGS.md)
A1_GDP = ("gdp-r1x", "gdp-r2x", "gdp-r4x", "gdp-r8x", "gdp-r16x")
A1_MP = ("a1-mp-r1x", "a1-mp-r2x", "a1-mp-r4x")
FRONTIER_RATIO = 0.001546
PILOT_GDP = {k: f"pilot-gdp-r{k}x" for k in (1, 2, 4, 8)}
PILOT_MP = {round(k * FRONTIER_RATIO, 6): f"pilot-mp-r{k}x" for k in (1, 2, 4, 8)}
PILOT_COMPOSITE = {r: f"pilot-comp-r{r}" for r in (0.0001, 0.000773, 0.003092, 0.006184, 0.012368)}
SAFEGUARD_T4, SAFEGUARD_GAP = 0.30, 0.02
GDP_LEVELS = {"gdp-r1x": 0.001546, "gdp-r2x": 0.003092, "gdp-r4x": 0.006184}
MP_LEVELS = {"mp-r2x": 0.003092, "mp-r4x": 0.006184, "mp-r8x": 0.012368}
COMPOSITE_LEVELS = {"comp-r0.000773": 0.000773, "comp-r0.003092": 0.003092, "comp-r0.006184": 0.006184}
CHECKPOINT_EVERY = 5  # the standard attacker's 11 checkpoints
REDUCTION = 0.05  # G-C1
NOT_TESTABLE = "not testable"


def reproduce_a1(gdp_root=REPO / "results/cia_v2/frontier/results", mp_root=REPO / "results/cia_v2/frontier/a1/results"):
    """A1's matched-accuracy difference at 0.740 (all candidates) from the committed runs, with its gate jackknife."""
    gdp = [load_audit(gdp_root, config=name) for name in A1_GDP]
    mp = [load_audit(mp_root, config=name) for name in A1_MP]
    difference = a1.matched_difference(mp, gdp, target=0.740, resamples=1)["difference"]
    return difference["estimate"], difference["low"], difference["high"]


def load_first_replicates(root, *, config, replicates):
    """Replicates ``0 .. replicates - 1`` of one configuration whose design has more replicates (stage a).

    Same arrays as ``load_audit``, but only the first ``replicates`` runs need to exist; refuses incomplete runs, mixed
    designs and inclusion that differs from the design's first columns."""
    found = {}
    for path in sorted(Path(root).glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest["config"] == config and manifest["replicate"] < replicates:
            if manifest["replicate"] in found:
                raise ValueError(f"{config!r}: replicate {manifest['replicate']} duplicated")
            found[manifest["replicate"]] = (path.parent, manifest)
    missing = [j for j in range(replicates) if j not in found or not (found[j][0] / "complete.json").exists()]
    if missing:
        raise ValueError(f"{config!r}: replicates {missing} missing or incomplete")
    designs = {(m["design_seed"], m["replicates"], tuple(m["checkpoint_rounds"]), tuple(m["candidates"]),
                tuple(m["top_decile"]), m.get("in_count")) for _, m in found.values()}
    if len(designs) != 1:
        raise ValueError(f"{config!r} mixes design settings: {sorted(designs)}")
    ((design_seed, total, rounds, candidates, top_decile, in_count),) = designs
    expected = design_matrix(total, design_seed=design_seed, in_count=in_count)[:, :replicates]
    losses = np.empty((len(candidates), replicates, len(rounds)))
    membership = np.zeros((len(candidates), replicates), dtype=bool)
    accuracy, client_minimum, names = np.empty(replicates), np.full(replicates, np.nan), []
    for j in range(replicates):
        folder, manifest = found[j]
        names.append(manifest["run_name"])
        index = {(r["round"], r["candidate"]): r for r in json.loads((folder / "measurements.json").read_text())}
        for c, candidate in enumerate(candidates):
            membership[c, j] = candidate in manifest["included"]
            losses[c, j] = [index[(t, candidate)]["shadow_loss"] for t in rounds]
        accuracy[j] = index[(rounds[-1], candidates[0])]["test_accuracy"]
        result = json.loads((folder / f"{manifest['run_name']}.json").read_text())
        clients = result.get("client_evaluate_metrics", {}).get(str(rounds[-1]), {})
        if "per-client-eval_acc" in clients:
            client_minimum[j] = min(map(float, clients["per-client-eval_acc"]))
    if not np.array_equal(membership, expected):
        raise ValueError(f"{config!r}: recorded inclusion differs from design seed {design_seed}")
    return Audit(config=config, rounds=tuple(rounds), losses=losses, membership=membership, final_accuracy=accuracy,
                 released_norms=None, surrogate_losses=None, run_names=tuple(names),
                 min_client_accuracy=None if np.isnan(client_minimum).all() else client_minimum,
                 candidates=tuple(candidates), top_decile=tuple(top_decile))


def g_p2a(summary):
    """G-P2a (STATE.md, Gate): vanilla all-candidate AUC point estimate at least 0.65 at R = 8."""
    estimate = summary["all"]["auc"]["estimate"]
    return {"threshold": G_P2A_THRESHOLD, "auc": estimate,
            "verdict": "pass" if estimate >= G_P2A_THRESHOLD else "fail: stop, ESCALATION.md, propose FEMNIST"}


def safeguard(t2, t4):
    """True when the pilot targets are informative (PROTOCOL.md Section 3); otherwise P2 stops before stage b."""
    return t4 >= SAFEGUARD_T4 and t2 - t4 >= SAFEGUARD_GAP


def choose_window(grid, t2, t4):
    """Three consecutive levels of ``grid`` ({ratio: accuracy}) for stage b (PROTOCOL.md Section 3).

    Among windows whose first level's accuracy is at least ``t2`` and last level's at most ``t4``, the smallest ratio;
    if none qualifies, the smallest shortfall ``max(0, t2 - first) + max(0, last - t4)``, ties to the smaller ratio."""
    levels = sorted(grid)
    windows = [levels[i:i + 3] for i in range(len(levels) - 2)]
    def shortfall(window):
        return max(0.0, t2 - grid[window[0]]) + max(0.0, grid[window[-1]] - t4)
    qualifying = [w for w in windows if shortfall(w) == 0]
    chosen = qualifying[0] if qualifying else min(windows, key=lambda w: (shortfall(w), w[0]))
    return {"levels": chosen, "qualifies": bool(qualifying), "shortfall": shortfall(chosen)}


def pilot_levels(root):
    """Pilot accuracies, targets, safeguard and the chosen windows."""
    def accuracy(config):
        return float(load_first_replicates(root, config=config, replicates=2).final_accuracy.mean())
    gdp = {k: accuracy(name) for k, name in PILOT_GDP.items()}
    mp = {r: accuracy(name) for r, name in PILOT_MP.items()}
    composite = {r: accuracy(name) for r, name in PILOT_COMPOSITE.items()}
    t2, t4 = gdp[2], gdp[4]
    out = {"accuracy": {"global-dp": {f"{k}x": a for k, a in gdp.items()},
                        "metric-privacy": {str(r): a for r, a in mp.items()},
                        "composite": {str(r): a for r, a in composite.items()}},
           "t2": t2, "t4": t4, "safeguard_passes": safeguard(t2, t4)}
    if out["safeguard_passes"]:
        out["metric-privacy"] = choose_window(mp, t2, t4)
        out["composite"] = choose_window(composite, t2, t4)
    return out


def standard_checkpoints(audit):
    """The audit at rounds 0, 5, ..., 50: the gated attackers' view of a run scored at every round."""
    keep = [i for i, t in enumerate(audit.rounds) if t % CHECKPOINT_EVERY == 0]
    return replace(audit, rounds=tuple(audit.rounds[i] for i in keep), losses=audit.losses[..., keep],
                   adjusted_losses=None if audit.adjusted_losses is None else audit.adjusted_losses[..., keep])


def _missing(value):
    return value is None or math.isnan(value)


def a1_outcome(entry):
    """G-A1's outcome at a target, with A1's "not bracketed" named as PROTOCOL.md Section 5 names it."""
    return NOT_TESTABLE if entry["outcome"] == "not bracketed" else entry["outcome"]


def c1_outcome(gate):
    """G-C1 at a target: at least 0.05 below metric privacy with the gate interval below 0, or not testable."""
    if any(_missing(gate[key]) for key in ("estimate", "low", "high")):
        return NOT_TESTABLE
    return "pass" if gate["estimate"] <= -REDUCTION and gate["high"] < 0 else "fail"


def a3_verdict(outcomes):
    """A-3 (STATE.md, Gate): equivalent or no detectable difference at every testable target."""
    testable = [o for o in outcomes if o != NOT_TESTABLE]
    if not testable:
        return NOT_TESTABLE
    fine = ("equivalent", "no detectable difference at this power")
    return "replicates" if all(o in fine for o in testable) else "does not replicate"


def d4_verdict(outcomes):
    """D-4 for one attacker (STATE.md, Gate): replicates, partial, does not replicate or not testable."""
    if "fail" in outcomes:
        return "does not replicate"
    if all(o == NOT_TESTABLE for o in outcomes):
        return NOT_TESTABLE
    return "replicates" if all(o == "pass" for o in outcomes) else "partial"


def lowest_client_at(audits, target):
    """Lowest-client accuracy at a target, linear in mean final accuracy; ``None`` outside the curve's range."""
    points = sorted((float(a.final_accuracy.mean()), float(np.mean(a.min_client_accuracy))) for a in audits)
    accuracy, lowest = np.array(points).T
    return float(np.interp(target, accuracy, lowest)) if accuracy[0] <= target <= accuracy[-1] else None


def scored(audit, losses=None):
    """Calibrated scores of one configuration, as the reanalysis' matched-accuracy statistic takes them."""
    losses = audit.losses if losses is None else losses
    return rz.Scored(calibrated_scores(losses, audit.membership), audit.membership, audit.final_accuracy)


def stage_b(root, resamples=m2.RESAMPLES):
    vanilla = load_audit(root, config="vanilla")
    gdp = [load_audit(root, config=name) for name in GDP_LEVELS]
    mp = [load_audit(root, config=name) for name in MP_LEVELS]
    every_round = [rz.load_config(root, name) for name in COMPOSITE_LEVELS]
    composite = [replace(c, audit=standard_checkpoints(c.audit)) for c in every_round]
    audits = [c.audit for c in composite]
    targets = tuple(float(gdp[i].final_accuracy.mean()) for i in (1, 2))  # a2, a4

    summaries = {"vanilla": m2.config_summary(vanilla, noise_ratio=0.0, resamples=resamples)}
    for levels, group in ((GDP_LEVELS, gdp), (MP_LEVELS, mp), (COMPOSITE_LEVELS, audits)):
        for (name, ratio), audit in zip(levels.items(), group):
            summaries[name] = m2.config_summary(audit, noise_ratio=ratio, resamples=resamples)
    sampling_aware = [rz.sampling_aware(c) for c in composite]
    for name, config, attacked, full in zip(COMPOSITE_LEVELS, composite, sampling_aware, every_round):
        everything = scored(full.audit)
        summaries[name]["realized_sampling_rate"] = rz.realized_rate(config)
        summaries[name]["sampling_aware_auc"] = m2._interval(
            pair_jackknife(attacked.scores, attacked.membership, roc_auc))
        summaries[name]["all_checkpoints_auc"] = m2._interval(
            pair_jackknife(everything.scores, everything.membership, roc_auc))

    g_a1 = a1.mechanism_report(mp, gdp, targets=targets, resamples=resamples)
    g_c1_standard = a1.mechanism_report(audits, mp, targets=targets, resamples=resamples)
    mp_scored = [scored(a) for a in mp]
    g_c1_aware = [rz.matched(sampling_aware, mp_scored, t, resamples=resamples) for t in targets]
    g_c1_everything = [rz.matched([scored(c.audit) for c in every_round], mp_scored, t, resamples=resamples)
                       for t in targets]

    a1_outcomes = [a1_outcome(g_a1["all"][str(t)]) for t in targets]
    standard_outcomes = [c1_outcome(g_c1_standard["all"][str(t)]["difference"]) for t in targets]
    aware_outcomes = [c1_outcome(r["gate"]) for r in g_c1_aware]
    fairness = {name: {str(t): lowest_client_at(group, t) for t in targets}
                for name, group in (("global-dp", gdp), ("metric-privacy", mp), ("composite", audits))}
    return {"targets": list(targets), "levels": a1._json(summaries),
            "g_a1": g_a1, "g_c1_standard": g_c1_standard, "g_c1_sampling_aware": a1._json({str(t): r for t, r in zip(targets, g_c1_aware)}),
            "reported_composite_all_checkpoints": a1._json({str(t): r for t, r in zip(targets, g_c1_everything)}),
            "lowest_client_accuracy_at_targets": fairness,
            "decisions": {"g_a1": dict(zip(map(str, targets), a1_outcomes)), "a3": a3_verdict(a1_outcomes),
                          "g_c1_standard": dict(zip(map(str, targets), standard_outcomes)),
                          "d4_standard": d4_verdict(standard_outcomes),
                          "g_c1_sampling_aware": dict(zip(map(str, targets), aware_outcomes)),
                          "d4_sampling_aware": d4_verdict(aware_outcomes)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("a", "pilot", "b"), required=True)
    parser.add_argument("--root", type=Path, default=HERE / "results")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    reproduced = reproduce_a1()
    print(json.dumps({"reproduction_a1_mp_minus_gdp_at_0.740": reproduced}))
    if tuple(round(x, 3) for x in reproduced) != A1_PUBLISHED:
        raise SystemExit(f"reproduction failed: {reproduced} != {A1_PUBLISHED}")
    if args.stage == "pilot":
        text = json.dumps(pilot_levels(HERE / "pilot"), indent=1, allow_nan=False)
        if args.out:
            args.out.write_text(text + "\n")
        print(text)
        return
    if args.stage == "b":
        out = {"reproduction_a1_mp_minus_gdp_at_0.740": list(reproduced), "stage": "b"} | stage_b(args.root)
        text = json.dumps(out, indent=1, allow_nan=False, default=float)
        if args.out:
            args.out.write_text(text + "\n")
        print(text)
        return
    audit = load_first_replicates(args.root, config="vanilla", replicates=STAGE_A_REPLICATES)
    summary = m2.config_summary(audit, noise_ratio=0.0)
    out = {"reproduction_a1_mp_minus_gdp_at_0.740": list(reproduced), "stage": "a",
           "replicates": STAGE_A_REPLICATES, "vanilla": a1._json(summary), "g_p2a": g_p2a(summary)}
    text = json.dumps(out, indent=1, allow_nan=False, default=float)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
