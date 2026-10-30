"""v3 re-analysis of committed v2 runs (PROTOCOL.md in this folder): sampling-aware attacker, early-stopping control,
per-candidate breakdown. Reads committed results only; launches nothing.

    PYTHONPATH="$PWD" uv run --no-sync python research/cia_roadmap_v3/reanalysis/reanalyze.py

Writes ``analysis.json`` next to this file. The attacker, the interval code and the pairs are the committed ones
(``experiments.cia.audit_attacks``, A1's matched-accuracy statistic); this module only builds new features and new
curves from them.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from experiments.cia.audit_attacks import (
    GATE_LEVEL, _complementary_pairs, _jackknife_interval, _resampled_indices, calibrated_scores, roc_auc,
)
from results.cia_v2.frontier import analyze as m2
from results.cia_v2.frontier.a1.analyze import curve_value
from results.cia_v2.randomized_inclusion.analysis import load_audit

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROOTS = {
    "gdp": REPO / "results/cia_v2/frontier/results",
    "a1": REPO / "results/cia_v2/frontier/a1/results",
    "b1": REPO / "results/cia_v2/mechanisms/b1_subsampling/results",
    "c1": REPO / "results/cia_v2/composite/results",
}
LAYOUT = next((REPO / "results/cia_frontier/consensus_kl/results/step3").glob("*/*-in-r0.001546*/partitions.json"))
GDP = ("gdp-r1x", "gdp-r2x", "gdp-r4x", "gdp-r8x", "gdp-r16x")  # A1's and C1's global-DP curve
GDP_WITH_VANILLA = ("vanilla",) + GDP  # early-stopping comparator (PROTOCOL.md (b))
MP = ("a1-mp-r1x", "a1-mp-r2x", "a1-mp-r4x")
B1_Q25 = ("b1-q0.25-r0.0001", "b1-q0.25-r0.000773", "b1-q0.25-r0.003092", "b1-q0.25-r0.006184")
B1_Q50 = ("b1-q0.5-r0.001546", "b1-q0.5-r0.003092", "b1-q0.5-r0.006184", "b1-q0.5-r0.012368")
C1 = ("c1-r0.000773", "c1-r0.003092", "c1-r0.006184", "c1-r0.012368")
TARGETS = (0.740, 0.660)
ES_TARGETS = (0.740, 0.700, 0.660)
TRUNCATIONS = (10, 15, 20)
SCHEDULE_SEED = 20261030
WINDOW = 5  # rounds between checkpoints
PUBLISHED_C1_VS_MP = {0.740: (-0.095, -0.137, -0.053), 0.660: (-0.108, -0.163, -0.053)}  # reports/cia_v2_composite.md


@dataclass
class Config:
    """One configuration: the committed audit plus per-checkpoint accuracy and (subsampled runs) schedules."""
    name: str
    audit: object
    accuracy_by_round: dict  # round -> (replicates,) server test accuracy at that checkpoint
    schedules: np.ndarray | None  # (candidates, replicates, rounds) bool, real for IN trials; None if full participation
    sampling_rate: float | None


def _runs(root, config):
    found = {}
    for path in sorted(Path(root).glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest["config"] == config:
            found[manifest["replicate"]] = (path.parent, manifest)
    return [found[j] for j in sorted(found)]


def load_config(root, config):
    audit = load_audit(root, config=config)
    runs = _runs(root, config)
    accuracy = {r: np.empty(len(runs)) for r in audit.rounds}
    rate = runs[0][1].get("sampling_rate")
    schedules = None
    if rate is not None:
        rounds = max(audit.rounds)
        schedules = np.zeros((len(audit.candidates), len(runs), rounds), dtype=bool)
    for j, (folder, manifest) in enumerate(runs):
        rows = json.loads((folder / "measurements.json").read_text())
        first = audit.candidates[0]
        for row in rows:
            if row["candidate"] == first:
                accuracy[row["round"]][j] = row["test_accuracy"]
        if schedules is not None:
            training = json.loads((folder / f"{manifest['run_name']}.json").read_text())["train_metrics"]
            position = {cid: c for c, cid in enumerate(audit.candidates)}
            for t in range(1, schedules.shape[2] + 1):
                for view_index in training[str(t)]["client-ids"]:
                    canonical = manifest["active"][int(view_index)]  # client-ids index the active view
                    if canonical in position:
                        schedules[position[canonical], j, t - 1] = True
    return Config(config, audit, accuracy, schedules, rate)


def simulated_schedules(config: Config) -> np.ndarray:
    """IN trials keep the real schedule; OUT trials get an i.i.d. Bernoulli(q) schedule (PROTOCOL.md (a))."""
    real = config.schedules
    out = real.copy()
    member = config.audit.membership
    for c, candidate in enumerate(config.audit.candidates):
        for j in range(real.shape[1]):
            if not member[c, j]:
                rng = np.random.default_rng([SCHEDULE_SEED, int(candidate), j])
                out[c, j] = rng.random(real.shape[2]) < config.sampling_rate
    return out


def schedule_statistic(losses: np.ndarray, schedules: np.ndarray, rate: float) -> np.ndarray:
    """``z = sum_w (n_w - 5q) (d_w - dbar_{c,w})`` over the checkpoint windows; (candidates, replicates).

    ``d_w = x_{w-1} - x_w`` is the window's drop in shadow loss and ``dbar_{c,w}`` the candidate's mean drop over all
    its replicates (no membership labels used), which removes the common loss decay (PROTOCOL.md amendment)."""
    c, r, rounds = schedules.shape
    counts = schedules.reshape(c, r, rounds // WINDOW, WINDOW).sum(axis=3)
    drops = losses[..., :-1] - losses[..., 1:]
    drops = drops - drops.mean(axis=1, keepdims=True)
    return ((counts - WINDOW * rate) * drops).sum(axis=2)


def realized_rate(config: Config) -> float:
    member = config.audit.membership
    return float(config.schedules[member].mean())


@dataclass
class Scored:
    """A curve point's inputs: scores, membership and per-replicate accuracy."""
    scores: np.ndarray
    membership: np.ndarray
    accuracy: np.ndarray


def standard(config: Config, subset=None, checkpoints=None) -> Scored:
    a = config.audit
    subset = np.ones(len(a.candidates), dtype=bool) if subset is None else subset
    k = len(a.rounds) if checkpoints is None else checkpoints
    losses = a.losses[subset][..., :k]
    accuracy = a.final_accuracy if checkpoints is None else config.accuracy_by_round[a.rounds[k - 1]]
    return Scored(calibrated_scores(losses, a.membership[subset]), a.membership[subset], accuracy)


def sampling_aware(config: Config, subset=None, z_only=False) -> Scored:
    a = config.audit
    subset = np.ones(len(a.candidates), dtype=bool) if subset is None else subset
    z = schedule_statistic(a.losses, simulated_schedules(config), config.sampling_rate)
    z = z / z.std()  # one constant per configuration, independent of membership
    features = z[..., None] if z_only else np.concatenate([a.losses, z[..., None]], axis=2)
    features = features[subset]
    return Scored(calibrated_scores(features, a.membership[subset]), a.membership[subset], a.final_accuracy)


def curve_at(points: list[Scored], target, index=None):
    values = []
    for p in points:
        scores, member, accuracy = p.scores, p.membership, p.accuracy
        if index is not None:
            scores, member, accuracy = scores[index], member[index], accuracy[index[1].ravel()]
        values.append((float(accuracy.mean()), roc_auc(scores.ravel(), member.ravel())))
    return curve_value(values, target)


def matched(mech: list[Scored], ref: list[Scored], target, resamples=m2.RESAMPLES, seed=m2.BOOTSTRAP_SEED):
    """A1's statistic on precomputed scores: pair jackknife at GATE_LEVEL and the two-way bootstrap."""
    candidates, replicates = mech[0].scores.shape
    rows = np.arange(candidates)[:, None]
    estimate = curve_at(mech, target) - curve_at(ref, target)
    pairs = _complementary_pairs(mech[0].membership)
    leave_out = [(rows, np.setdiff1d(np.arange(replicates), pair)[None, :]) for pair in pairs]
    values = np.array([curve_at(mech, target, i) - curve_at(ref, target, i) for i in leave_out])
    if math.isnan(estimate) or np.isnan(values).any():
        gate = {"estimate": estimate, "low": math.nan, "high": math.nan}
    else:
        iv = _jackknife_interval(estimate, values, GATE_LEVEL)
        gate = {"estimate": iv.estimate, "low": iv.low, "high": iv.high}
    draws = np.array([curve_at(mech, target, i) - curve_at(ref, target, i)
                      for i in _resampled_indices(candidates, replicates, resamples, seed)])
    finite = draws[np.isfinite(draws)]
    low, high = np.percentile(finite, [2.5, 97.5]) if len(finite) else (math.nan, math.nan)
    return {"target": target, "mechanism_auc": curve_at(mech, target), "reference_auc": curve_at(ref, target),
            "gate": gate, "two_way": {"low": float(low), "high": float(high), "bracketed": int(len(finite))}}


def paired_auc_difference(a: Scored, b: Scored):
    """AUC(a) - AUC(b) on the same trials, with the pair jackknife over replicates."""
    replicates = a.scores.shape[1]
    est = roc_auc(a.scores.ravel(), a.membership.ravel()) - roc_auc(b.scores.ravel(), b.membership.ravel())
    vals = []
    for pair in _complementary_pairs(a.membership):
        keep = np.setdiff1d(np.arange(replicates), pair)
        vals.append(roc_auc(a.scores[:, keep].ravel(), a.membership[:, keep].ravel())
                    - roc_auc(b.scores[:, keep].ravel(), b.membership[:, keep].ravel()))
    iv = _jackknife_interval(est, np.array(vals), GATE_LEVEL)
    return {"estimate": iv.estimate, "low": iv.low, "high": iv.high}


def decide_sa(result, standard_estimates):
    passes = all(r["gate"]["estimate"] <= -0.05 and r["gate"]["high"] < 0 for r in result)
    if not passes:
        return "lost"
    weakened = any(r["gate"]["estimate"] - standard_estimates[r["target"]] >= 0.03 for r in result)
    return "survives, weakened" if weakened else "survives"


def decide_es(by_t):
    """PROTOCOL.md (b) decision on B1 q = 0.25 against early-stopped global DP at T = 10 and 15."""
    bracketed = {t: [r for r in by_t[t] if not math.isnan(r["gate"]["low"])] for t in (10, 15)}
    if not any(bracketed.values()):
        return "not testable"
    for t, rows in bracketed.items():
        for r in rows:
            g = r["gate"]
            if g["low"] <= 0 <= g["high"] and g["low"] >= -0.05 and g["high"] <= 0.05:
                return "reducible to fewer contributions"
    specific = all(rows and all(r["gate"]["estimate"] <= -0.05 and r["gate"]["high"] < 0 for r in rows)
                   for rows in bracketed.values())
    return "specific to sampling" if specific else "inconclusive"


def per_candidate(points: list[Scored], target):
    """Per-candidate AUC interpolated in configuration mean accuracy."""
    candidates = points[0].scores.shape[0]
    out = np.empty(candidates)
    for c in range(candidates):
        vals = [(float(p.accuracy.mean()), roc_auc(p.scores[c], p.membership[c])) for p in points]
        out[c] = curve_value(vals, target)
    return out


def main():
    load = lambda root, names: [load_config(ROOTS[root], n) for n in names]  # noqa: E731
    gdp, gdp_v, mp = load("gdp", GDP), load("gdp", GDP_WITH_VANILLA), load("a1", MP)
    b25, b50, c1 = load("b1", B1_Q25), load("b1", B1_Q50), load("c1", C1)
    out = {"protocol": "research/cia_roadmap_v3/reanalysis/PROTOCOL.md", "inputs_commit": "c885c13f"}

    # Reproduction check before any new number (PROTOCOL.md, shared machinery).
    std = {k: [standard(c) for c in v] for k, v in {"gdp": gdp, "mp": mp, "b25": b25, "b50": b50, "c1": c1}.items()}
    repro = {t: matched(std["c1"], std["mp"], t) for t in TARGETS}
    for t, (e, lo, hi) in PUBLISHED_C1_VS_MP.items():
        g = repro[t]["gate"]
        if not (round(g["estimate"], 3) == e and round(g["low"], 3) == lo and round(g["high"], 3) == hi):
            raise SystemExit(f"reproduction failed at {t}: {g} against published {(e, lo, hi)}")
    out["reproduction_c1_vs_mp"] = {str(t): repro[t] for t in TARGETS}

    # (a) Sampling-aware attacker.
    out["realized_sampling_rate"] = {c.name: realized_rate(c) for c in b25 + b50 + c1}
    sa = {k: [sampling_aware(c) for c in v] for k, v in {"b25": b25, "b50": b50, "c1": c1}.items()}
    sz = {k: [sampling_aware(c, z_only=True) for c in v] for k, v in {"b25": b25, "b50": b50, "c1": c1}.items()}
    top = lambda cfg: np.isin(cfg.audit.candidates, cfg.audit.top_decile)  # noqa: E731
    c1_sa = [matched(sa["c1"], std["mp"], t) for t in TARGETS]
    std_est = {t: repro[t]["gate"]["estimate"] for t in TARGETS}
    out["sampling_aware"] = {
        "c1_vs_mp": c1_sa,
        "c1_vs_mp_top_decile": [matched([sampling_aware(c, top(c)) for c in c1], [standard(c, top(c)) for c in mp], t)
                                for t in TARGETS],
        "b1q25_vs_gdp": [matched(sa["b25"], std["gdp"], t) for t in TARGETS],
        "b1q25_vs_mp": [matched(sa["b25"], std["mp"], t) for t in TARGETS],
        "b1q25_vs_gdp_standard": [matched(std["b25"], std["gdp"], t) for t in TARGETS],
        "per_config": {c.name: {"standard_auc": roc_auc(s.scores.ravel(), s.membership.ravel()),
                                "sa_auc": roc_auc(a.scores.ravel(), a.membership.ravel()),
                                "z_only_auc": roc_auc(z.scores.ravel(), z.membership.ravel()),
                                "sa_minus_standard": paired_auc_difference(a, s),
                                "accuracy": float(c.audit.final_accuracy.mean())}
                       for k in ("b25", "b50", "c1")
                       for c, s, a, z in zip({"b25": b25, "b50": b50, "c1": c1}[k], std[k], sa[k], sz[k])},
    }
    out["sampling_aware"]["decision"] = decide_sa(c1_sa, std_est)

    # (b) Early-stopping control.
    es = {}
    for t in TRUNCATIONS:
        k = t // WINDOW + 1
        gdp_t = [standard(c, checkpoints=k) for c in gdp_v]
        mp_t = [standard(c, checkpoints=k) for c in mp]
        es[t] = {
            "gdp_curve": [(float(p.accuracy.mean()), roc_auc(p.scores.ravel(), p.membership.ravel())) for p in gdp_t],
            "mp_curve": [(float(p.accuracy.mean()), roc_auc(p.scores.ravel(), p.membership.ravel())) for p in mp_t],
            "b1q25_vs_es_gdp": [matched(std["b25"], gdp_t, a) for a in ES_TARGETS],
            "c1_vs_es_gdp": [matched(std["c1"], gdp_t, a) for a in ES_TARGETS],
        }
    out["early_stopping"] = {str(t): v for t, v in es.items()}
    out["early_stopping"]["decision"] = decide_es({t: es[t]["b1q25_vs_es_gdp"] for t in TRUNCATIONS})
    out["full_curves"] = {name: [(float(p.accuracy.mean()), roc_auc(p.scores.ravel(), p.membership.ravel()))
                                 for p in std[name]] for name in std}
    out["full_curves"]["c1_sa"] = [(float(p.accuracy.mean()), roc_auc(p.scores.ravel(), p.membership.ravel()))
                                   for p in sa["c1"]]
    out["full_curves"]["b25_sa"] = [(float(p.accuracy.mean()), roc_auc(p.scores.ravel(), p.membership.ravel()))
                                    for p in sa["b25"]]

    # (c) Per-candidate breakdown.
    layout = {c["target"]: c for c in json.loads(LAYOUT.read_text())["clients"]}
    cands = c1[0].audit.candidates
    records = np.array([layout[c]["records"] for c in cands], dtype=float)
    def entropy(counts):
        p = np.asarray(counts, dtype=float); p = p[p > 0] / p.sum()
        return float(-(p * np.log(p)).sum())
    ent = np.array([entropy(layout[c]["class_counts"]) for c in cands])
    pc = {}
    for t in TARGETS:
        delta = per_candidate(std["c1"], t) - per_candidate(std["mp"], t)
        rho_r, p_r = spearmanr(delta, records)
        rho_e, p_e = spearmanr(delta, ent)
        pc[str(t)] = {"delta": delta.tolist(), "median": float(np.median(delta)),
                      "fraction_negative": float((delta < 0).mean()),
                      "spearman_records": [float(rho_r), float(p_r)], "spearman_entropy": [float(rho_e), float(p_e)]}
    out["per_candidate"] = {"candidates": list(cands), "records": records.tolist(), "label_entropy": ent.tolist(), **pc}

    (HERE / "analysis.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({"reproduced": True, "sa_decision": out["sampling_aware"]["decision"],
                      "es_decision": out["early_stopping"]["decision"]}))


if __name__ == "__main__":
    main()
