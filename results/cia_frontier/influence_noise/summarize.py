"""Descriptive per-arm summary of collected pilot trajectories (no training, no claims).

Round-100 paired IN-vs-OUT target losses (lower = more IN-like), accuracies,
per-class recall, matched-energy checks, and Spearman correlations between each
target's IN-run influence/weighted-norm diagnostics and its loss gap. One seed,
10 correlated targets: no intervals, no significance, no privacy certification.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _trajectory(folder: Path):
    (complete,) = folder.glob("*/complete.json")
    traj = complete.parent
    (train,) = [p for p in traj.glob("*.json") if p.name.endswith("__eurosat_cnn.json")]
    (evaluation,) = traj.glob("*.evaluation.json")
    return (json.loads((traj / "measurements.json").read_text()), json.loads(train.read_text()),
            json.loads(evaluation.read_text()), json.loads((traj / "influence_protocol.json").read_text()))


def _rank(x):
    x = np.asarray(x, float)
    order = x.argsort()
    ranks = np.empty(len(x))
    ranks[order] = np.arange(len(x))
    for value in np.unique(x):  # average ties
        ranks[x == value] = ranks[x == value].mean()
    return ranks


def spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize_arm(root: Path, fraction: float, seed: int = 42, targets=range(10), final=100):
    tag = f"f{fraction!r}-seed-{seed}"
    meas_in, train_in, eval_in, proto_in = _trajectory(root / f"{tag}-in")
    rows, recall = [], {}
    in_final = {r["target"]: r for r in meas_in if r["round"] == final}
    tm = train_in["train_metrics"]
    rounds = sorted(tm, key=int)
    energy = np.array([tm[r]["influence-realized-noise-sq-norm"] for r in rounds])
    expected = tm[rounds[0]]["influence-expected-noise-sq-norm"]
    for t in targets:
        meas_out, train_out, eval_out, proto_out = _trajectory(root / f"{tag}-out-{t}")
        assert proto_out["fraction"] == proto_in["fraction"] == fraction
        (out_final,) = [r for r in meas_out if r["round"] == final and r["target"] == t]
        idx = [tm[r]["influence-client-ids"].index(t) for r in rounds]
        infl = np.array([tm[r]["influence-raw-norms"][i] for r, i in zip(rounds, idx)])
        wnorm = np.array([tm[r]["influence-weighted-clipped-norms"][i] for r, i in zip(rounds, idx)])
        dstd = np.array([tm[r]["influence-directional-noise-stdv"][i] for r, i in zip(rounds, idx)])
        rows.append({
            "target": t,
            "clean_in": in_final[t]["clean_loss"], "clean_out": out_final["clean_loss"],
            "noisy_in": in_final[t]["noisy_loss"], "noisy_out": out_final["noisy_loss"],
            "clean_gap_out_minus_in": out_final["clean_loss"] - in_final[t]["clean_loss"],
            "noisy_gap_out_minus_in": out_final["noisy_loss"] - in_final[t]["noisy_loss"],
            "out_accuracy": eval_out["server_final_test"]["accuracy"],
            "in_mean_influence_norm": float(infl.mean()),
            "in_mean_weighted_clipped_norm": float(wnorm.mean()),
            "in_mean_directional_stdv_over_tau": float(dstd.mean() / proto_in["tau"]),
            "weight": tm[rounds[0]]["influence-weights"][idx[0]],
        })
        recall[t] = [c["recall"] for c in eval_out["server_final_test"]["per_class"].values()]
    gaps = [r["clean_gap_out_minus_in"] for r in rows]
    noisy_gaps = [r["noisy_gap_out_minus_in"] for r in rows]
    per_class_in = eval_in["server_final_test"]["per_class"]
    return {
        "fraction": fraction, "seed": seed, "round": final, "targets": list(targets),
        "in_accuracy": eval_in["server_final_test"]["accuracy"],
        "out_accuracy_mean": float(np.mean([r["out_accuracy"] for r in rows])),
        "out_accuracy_range": [float(min(r["out_accuracy"] for r in rows)), float(max(r["out_accuracy"] for r in rows))],
        "in_per_class_recall": {v["name"]: v["recall"] for v in per_class_in.values()},
        "out_per_class_recall_mean": dict(zip([v["name"] for v in per_class_in.values()],
                                              np.mean(list(recall.values()), axis=0).tolist())),
        "clean_in_lower_count": int(sum(g > 0 for g in gaps)),
        "noisy_in_lower_count": int(sum(g > 0 for g in noisy_gaps)),
        "clean_gap_mean": float(np.mean(gaps)), "noisy_gap_mean": float(np.mean(noisy_gaps)),
        "energy_realized_over_expected_mean": float(energy.mean() / expected),
        "energy_realized_over_expected_sd": float(energy.std() / expected),
        "expected_noise_sq_norm": expected,
        "fallback_rounds": int(sum(tm[r]["influence-fallback-isotropic"] for r in rounds)),
        "cap_bound_client_rounds": int(sum(sum(tm[r]["influence-cap-bound"]) for r in rounds)),
        "spearman_clean_gap_vs": {k: spearman([r[k] for r in rows], gaps) for k in
                                  ("in_mean_influence_norm", "in_mean_weighted_clipped_norm",
                                   "in_mean_directional_stdv_over_tau", "weight")},
        "spearman_noisy_gap_vs": {k: spearman([r[k] for r in rows], noisy_gaps) for k in
                                  ("in_mean_influence_norm", "in_mean_weighted_clipped_norm",
                                   "in_mean_directional_stdv_over_tau", "weight")},
        "per_target": rows,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.0, 0.5])
    args = parser.parse_args(argv)
    print(json.dumps([summarize_arm(args.root, f) for f in args.fractions], indent=2))


if __name__ == "__main__":
    main()
