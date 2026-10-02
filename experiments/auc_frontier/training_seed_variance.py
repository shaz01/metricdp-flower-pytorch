"""Split per-client attack-score variance into client differences vs training randomness.

Data layout (partitions, splits, shadow sets) is fixed by partition seed 42 in every run.
Training seed 42 is the existing pinned-runtime vanilla panel (``frontier_torch210``);
training seeds 43/44 come from ``training_seed_variance`` (``--partition-seed 42``).

Per-client score: fraction of rounds 1..100 where IN clean-shadow loss < OUT clean-shadow
loss for that client (no direction flip). ``score(c, i, j)`` pairs IN from training seed i
with OUT from training seed j; all pairings are valid because the data layout is identical.

Usage: uv run python -m experiments.auc_frontier.training_seed_variance
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path("results/new_auc_frontier_eurosat")
SOURCES = {42: ROOT / "frontier_torch210", 43: ROOT / "training_seed_variance",
           44: ROOT / "training_seed_variance"}
TARGETS = (0, 1, 2, 3, 4)
ROUNDS = range(1, 101)


def load():
    runs = {}
    for seed, root in SOURCES.items():
        for path in root.rglob("manifest.json"):
            m = json.loads(path.read_text())
            if m["privacy"] != "vanilla" or m["seed"] != seed or m.get("alpha") != 0.3:
                continue
            if m.get("partition_seed", m["seed"]) != 42:
                continue
            if m["out_target"] is not None and m["out_target"] not in TARGETS:
                continue
            if not (path.parent / "complete.json").exists():
                raise ValueError(f"incomplete: {path.parent}")
            key = (seed, m["out_target"])
            if key in runs:
                raise ValueError(f"duplicate trajectory {key}")
            rows = json.loads((path.parent / "measurements.json").read_text())
            runs[key] = {(r["round"], r["target"]): r["clean_loss"] for r in rows}
    missing = [(s, t) for s in SOURCES for t in (None, *TARGETS) if (s, t) not in runs]
    if missing:
        raise ValueError(f"missing trajectories: {missing}")
    return runs


def score(runs, client, in_seed, out_seed):
    IN, OUT = runs[(in_seed, None)], runs[(out_seed, client)]
    return st.fmean(1.0 if IN[(r, client)] < OUT[(r, client)] else 0.0 for r in ROUNDS)


def one_way(groups):
    """Random-effects one-way ANOVA: variance between groups vs within groups."""
    k, n = len(groups), len(groups[0])
    grand = st.fmean(v for g in groups for v in g)
    msb = n * sum((st.fmean(g) - grand) ** 2 for g in groups) / (k - 1)
    msw = sum((v - st.fmean(g)) ** 2 for g in groups for v in g) / (k * (n - 1))
    return max((msb - msw) / n, 0.0), msw


def main():
    runs = load()
    seeds = sorted(SOURCES)
    matched = {c: [score(runs, c, s, s) for s in seeds] for c in TARGETS}
    client_var, training_var = one_way([matched[c] for c in TARGETS])
    # IN vs OUT randomness: 3x3 table per client (rows IN seed, cols OUT seed), no replication.
    in_vars, out_vars, resid = [], [], []
    for c in TARGETS:
        table = [[score(runs, c, i, j) for j in seeds] for i in seeds]
        grand = st.fmean(v for row in table for v in row)
        rows = [st.fmean(row) for row in table]
        cols = [st.fmean(table[i][j] for i in range(3)) for j in range(3)]
        ms_row = 3 * sum((x - grand) ** 2 for x in rows) / 2
        ms_col = 3 * sum((x - grand) ** 2 for x in cols) / 2
        ms_res = sum((table[i][j] - rows[i] - cols[j] + grand) ** 2
                     for i in range(3) for j in range(3)) / 4
        in_vars.append(max((ms_row - ms_res) / 3, 0.0))
        out_vars.append(max((ms_col - ms_res) / 3, 0.0))
        resid.append(ms_res)
    result = dict(
        matched_scores={str(c): v for c, v in matched.items()},
        client_variance=client_var, training_variance=training_var,
        client_share=client_var / (client_var + training_var) if client_var + training_var else None,
        in_randomness_variance=st.fmean(in_vars), out_randomness_variance=st.fmean(out_vars),
        interaction_residual_variance=st.fmean(resid),
        note="5 clients x 3 training seeds, vanilla, one partition seed: estimates are rough.",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
