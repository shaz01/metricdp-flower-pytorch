"""Correlate each influence definition with the per-client attack score.

For every (dataset, arm, seed) with an IN run and OUT runs: the attack score per client
comes from per_client_score.py (fraction of rounds where the client's clean-shadow loss is
lower under IN than under its OUT; 0.5 = no signal). The influence features come from the IN
run JSON's ``influence-*`` train metrics (aggregated over rounds) plus the partition summary.
Prints Spearman and Pearson correlations per feature, pooled and per setting, and writes
correlations.json next to the results.

    python -m results.cia_frontier.influence_correlation.analyze [root ...] [--out-dir DIR --tag _x]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

from results.cia_frontier.per_client_score import score_directories

PREFIX = "influence-"
# feature name -> (train-metric suffix, how to aggregate the per-round values)
LOGGED = {
    "update_norm_mean": ("update-norm-before-clipping", "mean"),
    "update_norm_sum": ("update-norm-before-clipping", "sum"),
    "clipped_fraction": ("clipped", "mean"),
    "distance_from_average_mean": ("distance-from-weighted-clipped-average", "mean"),
    "cosine_with_average_mean": ("cosine-with-weighted-clipped-average", "mean"),
    "leave_one_out_influence_mean": ("leave-one-out-influence-norm", "mean"),
    "leave_one_out_influence_sum": ("leave-one-out-influence-norm", "sum"),
    "aggregation_weight": ("aggregation-weight", "mean"),
    "train_loss_mean": (None, "mean"),  # from per-client-train_loss
}


def _agg(values, how):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return None
    return statistics.fmean(values) if how == "mean" else sum(values)


# Pairwise client-model distance (metric privacy's d): logged as influence-pairwise-* for every
# privacy mode since seeds 43/44; older metric-privacy runs only have metric-dp-pairwise-* (the IN
# runs score all clients, so its local client IDs are the canonical ones).
PAIRWISE_SOURCES = (("influence-pairwise-distances", "influence-pairwise-client-i", "influence-pairwise-client-j"),
                    ("metric-dp-pairwise-distances", "metric-dp-pairwise-client-i", "metric-dp-pairwise-client-j"))
DISTANCE_FEATURES = ("pairwise_distance_mean", "pairwise_distance_max", "max_pair_share")


def pairwise_round(metrics: dict) -> dict[int, dict[str, float]] | None:
    """One round: per client, mean and max distance to the others and whether it is in the max pair."""
    for dist_key, i_key, j_key in PAIRWISE_SOURCES:
        if metrics.get(dist_key):
            dists, ci, cj = metrics[dist_key], metrics[i_key], metrics[j_key]
            break
    else:
        return None
    to_others: dict[int, list[float]] = defaultdict(list)
    for d, a, b in zip(dists, ci, cj, strict=True):
        to_others[int(a)].append(float(d))
        to_others[int(b)].append(float(d))
    top = max(range(len(dists)), key=lambda k: dists[k])
    in_max = {int(ci[top]), int(cj[top])}
    return {c: {"pairwise_distance_mean": statistics.fmean(v), "pairwise_distance_max": max(v),
                "max_pair_share": 1.0 if c in in_max else 0.0} for c, v in to_others.items()}


def client_features(run_json: dict, partition_summary: dict | None) -> dict[int, dict]:
    """Per canonical client: aggregated influence features over all logged rounds.

    Distance features (only where a pairwise matrix was logged), averaged over rounds:
    ``pairwise_distance_mean`` = mean distance to the other clients, ``pairwise_distance_max`` =
    max distance to the others, ``max_pair_share`` = share of rounds the client is in the max
    pair (i.e. sets metric privacy's d).
    """
    train = run_json.get("train_metrics", {})
    per_client: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for metrics in train.values():
        for cid, values in (pairwise_round(metrics) or {}).items():
            for name, value in values.items():
                per_client[cid][name].append(value)
        ids = metrics.get(PREFIX + "client-ids")
        if ids:
            for name, (suffix, _) in LOGGED.items():
                if suffix is None:
                    continue
                for cid, value in zip(ids, metrics.get(PREFIX + suffix, []), strict=True):
                    per_client[int(cid)][name].append(float(value))
        loss_ids, losses = metrics.get("client-ids"), metrics.get("per-client-train_loss")
        if loss_ids and losses:
            for cid, value in zip(loss_ids, losses, strict=True):
                per_client[int(cid)]["train_loss_mean"].append(float(value))
    sizes = {}
    if partition_summary:
        sizes = {c["target"]: c["train_records"] for c in partition_summary["clients"]}
    out = {}
    for cid, series in per_client.items():
        row = {name: _agg(series.get(name, []), how) for name, (_, how) in LOGGED.items()}
        row.update({name: _agg(series[name], "mean") for name in DISTANCE_FEATURES if series.get(name)})
        if cid in sizes:
            row["train_records"] = sizes[cid]
        out[cid] = row
    return out


def label_skew(partition_summary: dict | None) -> dict[int, float]:
    """Total-variation distance of each client's class mix from the pooled mix, if recorded."""
    if not partition_summary or "class_counts" not in partition_summary["clients"][0]:
        return {}
    counts = {c["target"]: Counter(c["class_counts"]) for c in partition_summary["clients"]}
    pooled = sum(counts.values(), Counter())
    total = sum(pooled.values())
    out = {}
    for cid, cnt in counts.items():
        n = sum(cnt.values()) or 1
        out[cid] = 0.5 * sum(abs(cnt[k] / n - pooled[k] / total) for k in pooled)
    return out


def _rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def pearson(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def spearman(x, y):
    return pearson(_rank(x), _rank(y))


def collect(root: Path) -> list[dict]:
    """One row per (dataset, arm, seed, client) with attack score and features."""
    rows = []
    for scored in score_directories(root):
        in_folder = next(p.parent for p in root.rglob("manifest.json")
                         if (m := json.loads(p.read_text()))["out_target"] is None
                         and m["seed"] == scored["seed"] and m["privacy"] == scored["privacy"]
                         and m.get("dataset") == scored.get("dataset")
                         and m.get("noise_ratio") == scored.get("noise_ratio"))
        manifest = json.loads((in_folder / "manifest.json").read_text())
        run_json = in_folder / f"{manifest['run_name']}.json"  # the training runner's result
        data = json.loads(run_json.read_text())
        summary_path = in_folder / "partitions.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else None
        features = client_features(data, summary)
        skew = label_skew(summary)
        for target, score in scored["per_target"].items():
            cid = int(target)
            rows.append({"dataset": scored.get("dataset"), "privacy": scored["privacy"],
                         "noise_ratio": scored.get("noise_ratio"), "seed": scored["seed"],
                         "client": cid, "attack_score": score,
                         **features.get(cid, {}), **({"label_skew": skew[cid]} if cid in skew else {})})
    return rows


def correlations(rows: list[dict]) -> dict:
    features = sorted({k for r in rows for k in r} - {"dataset", "privacy", "noise_ratio", "seed",
                                                        "client", "attack_score"})
    def table(subset):
        out = {}
        for f in features:
            pairs = [(r[f], r["attack_score"]) for r in subset if r.get(f) is not None]
            if len(pairs) >= 3:
                x, y = zip(*pairs)
                out[f] = {"n": len(pairs), "spearman": spearman(x, y), "pearson": pearson(x, y)}
        return out
    by_setting, by_seed = defaultdict(list), defaultdict(list)
    for r in rows:
        by_setting[f"{r['dataset']}|{r['privacy']}|{r['noise_ratio']}"].append(r)
        by_seed[f"{r['dataset']}|{r['privacy']}|{r['noise_ratio']}|seed{r['seed']}"].append(r)
    out = {"pooled": table(rows), "n_rows": len(rows), "seeds": sorted({r["seed"] for r in rows}),
           "per_setting": {k: table(v) for k, v in sorted(by_setting.items())}}
    if len(out["seeds"]) > 1:
        out["per_setting_seed"] = {k: table(v) for k, v in sorted(by_seed.items())}
        out["distance_feature_seeds"] = {
            k: sorted({r["seed"] for r in v if r.get("pairwise_distance_mean") is not None})
            for k, v in sorted(by_setting.items())}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", type=Path, nargs="*",
                        default=[Path("results/cia_frontier/influence_correlation/results")])
    parser.add_argument("--out-dir", type=Path, help="Where to write the JSONs (default: first root)")
    parser.add_argument("--tag", default="", help="Suffix for the output names, e.g. _s42_44")
    args = parser.parse_args()
    rows = [row for root in args.roots for row in collect(root)]
    result = correlations(rows)
    out_dir = args.out_dir or args.roots[0]
    (out_dir / f"correlations{args.tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / f"client_rows{args.tag}.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(f"{len(rows)} (client, setting) rows\n")
    print(f"{'feature':<32}{'n':>5}{'spearman':>10}{'pearson':>9}")
    for f, v in sorted(result["pooled"].items(), key=lambda kv: -abs(kv[1]["spearman"] or 0)):
        fmt = lambda x: "   -" if x is None else f"{x:+.2f}"
        print(f"{f:<32}{v['n']:>5}{fmt(v['spearman']):>10}{fmt(v['pearson']):>9}")


if __name__ == "__main__":
    main()
