"""Size-weighted vs equal-weighted FedAvg: does client size still predict exposure?

Per setting (dataset, arm) and per root: Spearman of client size (train_records) vs
  - the attack score (per_client_score: fraction of rounds IN clean loss < OUT clean loss), and
  - the mean OUT-IN clean-shadow-loss gap over matched rounds (larger = more exposed).
Writes ``weighting_comparison.json`` into the equal-weight root and prints a side-by-side table.

    python -m results.cia_frontier.influence_correlation.compare_weighting \
        [--size-weighted ROOT] [--equal ROOT]
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from results.cia_frontier.influence_correlation.analyze import collect, spearman


def loss_gaps(root: Path) -> dict:
    """{(dataset, privacy, noise_ratio, seed): {target: mean(OUT - IN clean loss)}}."""
    ins, outs = {}, defaultdict(dict)
    for path in sorted(root.rglob("manifest.json")):
        m = json.loads(path.read_text())
        key = (m.get("dataset"), m["privacy"], m.get("noise_ratio"), m["seed"])
        rows = {(r["round"], r["target"]): float(r["clean_loss"])
                for r in json.loads((path.parent / "measurements.json").read_text())}
        if m["out_target"] is None:
            ins[key] = rows
        else:
            outs[key][m["out_target"]] = rows
    gaps = {}
    for key, by_target in outs.items():
        gaps[key] = {t: statistics.fmean(v - ins[key][k] for k, v in rows.items() if k[1] == t)
                     for t, rows in by_target.items()}
    return gaps


def per_setting(root: Path) -> dict:
    rows, gaps = collect(root), loss_gaps(root)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["privacy"], r["noise_ratio"], r["seed"])].append(r)
    out = {}
    for key, group in sorted(groups.items()):
        group.sort(key=lambda r: r["client"])
        size = [r["train_records"] for r in group]
        score = [r["attack_score"] for r in group]
        gap = [gaps[key][r["client"]] for r in group]
        out[f"{key[0]}|{key[1]}"] = {
            "n": len(group), "mean_attack_score": statistics.fmean(score),
            "mean_loss_gap": statistics.fmean(gap),
            "spearman_size_vs_score": spearman(size, score),
            "spearman_size_vs_loss_gap": spearman(size, gap),
        }
    return out


def main():
    base = Path("results/cia_frontier/influence_correlation")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-weighted", type=Path, default=base / "results")
    parser.add_argument("--equal", type=Path, default=base / "results_eqw")
    args = parser.parse_args()
    sw, eq = per_setting(args.size_weighted), per_setting(args.equal)
    result = {"size_weighted": sw, "equal": eq,
              "note": "Spearman over the 16 clients of each setting (seed 42); gap = mean over "
                      "rounds of OUT minus IN clean-shadow loss"}
    (args.equal / "weighting_comparison.json").write_text(json.dumps(result, indent=1) + "\n")
    def pair(key, fmt):
        cells = []
        for d in (sw.get(k, {}), eq.get(k, {})):
            v = d.get(key)
            cells.append("-" if v is None else format(v, fmt))
        return " / ".join(cells)

    print(f"{'setting (SW / EQ)':<26}{'rho(size,score)':>17}{'rho(size,gap)':>16}"
          f"{'mean score':>16}{'mean gap':>20}")
    for k in sorted(set(sw) | set(eq)):
        print(f"{k:<26}{pair('spearman_size_vs_score', '+.2f'):>17}"
              f"{pair('spearman_size_vs_loss_gap', '+.2f'):>16}{pair('mean_attack_score', '.3f'):>16}"
              f"{pair('mean_loss_gap', '+.4f'):>20}")


if __name__ == "__main__":
    main()
