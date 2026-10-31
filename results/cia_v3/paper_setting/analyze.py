"""P1 analysis: the paper's 3-client setting under measurement standard v3 (PROTOCOL.md Section 3).

Per configuration, the 40 worlds (pair 0 IN, pair 0 OUT, pair 1 IN, ...) become arrays with one candidate, the target:
losses ``(1, 40, 20)`` at rounds 1-20 and membership ``(1, 40)``. Reported: the paper's pooled-checkpoint AUC per pair
(Table 13's statistic), v2's calibrated attacker over trajectories with a 95% pair jackknife, the threshold attacker,
Table 13's accuracy, and G-P1 as fixed in PROTOCOL.md. The reproduction check (M2's vanilla AUC 0.723) runs first.

    uv run --no-sync python -m results.cia_v3.paper_setting.analyze --main vanilla gdp-z0.01 mp-z0.01 \
        --out results/cia_v3/paper_setting/analysis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cia.audit_attacks import calibrated_scores, pair_jackknife, roc_auc, threshold_scores, tpr_at_fpr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ROUNDS = tuple(range(1, 21))  # the paper's statistic and the attacker use rounds 1-20 (round 0 is the shared init)
LEVEL = 0.95
ACCURACY_ROUNDS = (16, 17, 18, 19, 20)  # Table 13: mean over the last 5 rounds
M2_VANILLA_AUC = 0.723  # reports/cia_v2_m2_frontier.md, all candidates


def reproduce_m2(root=REPO / "results/cia_v2/frontier/results"):
    """M2's vanilla all-candidate AUC from the committed runs, with the attacker P1 uses (standard v3 item 7)."""
    from results.cia_v2.randomized_inclusion.analysis import load_audit

    audit = load_audit(root, config="vanilla")
    return roc_auc(calibrated_scores(audit.losses, audit.membership), audit.membership)


def load_config(root, config):
    """``(losses, clean_losses, membership, worlds)`` of one configuration; refuses incomplete or mixed designs."""
    found = {}
    for path in sorted(Path(root).glob("*/manifest.json")):
        manifest = json.loads(path.read_text())
        if manifest["config"] != config:
            continue
        if not (path.parent / "complete.json").exists():
            raise ValueError(f"{config!r}: incomplete run {manifest['run_name']}")
        found[(manifest["pair"], manifest["world"])] = (path.parent, manifest)
    if not found:
        raise ValueError(f"no runs of {config!r} under {root}")
    designs = {(m["design_seed"], m["pairs"], m["noise_multiplier"], m["privacy"]) for _, m in found.values()}
    if len(designs) != 1:
        raise ValueError(f"{config!r} mixes design settings: {sorted(designs)}")
    pairs = sorted({pair for pair, _ in found})
    missing = [p for p in pairs if (p, "in") not in found or (p, "out") not in found]
    if missing:
        raise ValueError(f"{config!r}: pairs {missing} lack a world")
    noisy, clean, membership, worlds = [], [], [], []
    for pair in pairs:
        for world in ("in", "out"):
            folder, manifest = found[(pair, world)]
            rows = {r["round"]: r for r in json.loads((folder / "measurements.json").read_text())}
            noisy.append([rows[r]["noisy_shadow_loss"] for r in ROUNDS])
            clean.append([rows[r]["shadow_loss"] for r in ROUNDS])
            membership.append(world == "in")
            worlds.append((folder, manifest))
    return np.array(noisy)[None], np.array(clean)[None], np.array(membership)[None], worlds


def paper_statistic(losses, membership):
    """Table 13's AUC per pair: 20 IN round scores ``-loss`` against 20 OUT round scores."""
    values = []
    for p in range(losses.shape[1] // 2):
        inside, outside = losses[0, 2 * p], losses[0, 2 * p + 1]
        if not (membership[0, 2 * p] and not membership[0, 2 * p + 1]):
            raise ValueError("worlds must be ordered IN, OUT within each pair")
        values.append(roc_auc(-np.concatenate([inside, outside]),
                              np.r_[np.ones(len(inside), bool), np.zeros(len(outside), bool)]))
    return np.array(values)


def _interval(result):
    return {"estimate": result.estimate, "low": result.low, "high": result.high}


def table13_accuracy(worlds):
    """Aggregated client-test accuracy averaged over rounds 16-20, and the lowest client's at round 20, per world."""
    mean, lowest = [], []
    for folder, manifest in worlds:
        metrics = json.loads((folder / f"{manifest['run_name']}.json").read_text())["client_evaluate_metrics"]
        mean.append(float(np.mean([metrics[str(r)]["eval_acc"] for r in ACCURACY_ROUNDS])))
        lowest.append(min(map(float, metrics[str(ACCURACY_ROUNDS[-1])]["per-client-eval_acc"])))
    return np.array(mean), np.array(lowest)


def _spread(values):
    return {"mean": float(np.mean(values)), "min": float(np.min(values)), "max": float(np.max(values))}


def summarize(losses, clean, membership, accuracy=None, lowest=None):
    paper = paper_statistic(losses, membership)
    scores = calibrated_scores(losses, membership)
    member = membership
    out = {
        "pairs": int(losses.shape[1] // 2),
        "paper_statistic": {"per_pair": paper.tolist(), "median": float(np.median(paper)),
                            "p05": float(np.percentile(paper, 5)), "p95": float(np.percentile(paper, 95)),
                            "below_0.5": int((paper < 0.5).sum())},
        "calibrated": _interval(pair_jackknife(scores, member, roc_auc, level=LEVEL)),
        "calibrated_clean": _interval(pair_jackknife(calibrated_scores(clean, member), member, roc_auc, level=LEVEL)),
        "threshold": _interval(pair_jackknife(threshold_scores(losses), member, roc_auc, level=LEVEL)),
        "tpr_at_5pct_fpr": tpr_at_fpr(scores, member, 0.05),
    }
    if accuracy is not None:
        inside = member[0]
        out["accuracy"] = {"in": _spread(accuracy[inside]), "out": _spread(accuracy[~inside])}
        out["lowest_client_accuracy"] = {"in": _spread(lowest[inside]), "out": _spread(lowest[~inside])}
    return out


def decide(summaries, gdp, mp):
    """G-P1 (PROTOCOL.md Section 3 item 4) on the two DP configurations."""
    statements = []
    if all(summaries[c]["calibrated"]["low"] > 0.5 for c in (gdp, mp)):
        statements.append("the published protection claim fails in its own setting")
    if any(summaries[c]["paper_statistic"]["p05"] < 0.5 < summaries[c]["paper_statistic"]["p95"] for c in (gdp, mp)):
        statements.append("Table 13 is a single draw from a wide distribution")
    return statements or ["consistent with the paper"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=HERE / "results")
    parser.add_argument("--main", nargs=3, metavar=("VANILLA", "GDP", "MP"), required=True)
    parser.add_argument("--extension", nargs=2, metavar=("GDP", "MP"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    reproduced = reproduce_m2()
    print(json.dumps({"reproduction_m2_vanilla_auc": reproduced}))
    if round(reproduced, 3) != M2_VANILLA_AUC:
        raise SystemExit(f"reproduction failed: {reproduced:.4f} != {M2_VANILLA_AUC}")
    configs = list(args.main) + list(args.extension or [])
    summaries = {}
    for config in configs:
        losses, clean, membership, worlds = load_config(args.root, config)
        accuracy, lowest = table13_accuracy(worlds)
        summaries[config] = summarize(losses, clean, membership, accuracy, lowest)
    out = {"reproduction_m2_vanilla_auc": reproduced, "level": LEVEL, "rounds": list(ROUNDS), "configs": summaries,
           "g_p1": decide(summaries, args.main[1], args.main[2])}
    if args.extension:
        out["g_p1_extension"] = decide(summaries, *args.extension)
    text = json.dumps(out, indent=1)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
