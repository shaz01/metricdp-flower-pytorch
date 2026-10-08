"""Stage C side by side: size-weighted FedAvg (stage_c, R=50) vs equal-weighted (stage_c_eqw, R=57).

Reads each root's ``frontier.json`` (run ``frontier --root`` first) and writes
``weighting_comparison.json`` into the equal-weight root; prints a table of round-R accuracy
and mean per-client attack score per setting.

    python -m results.cia_frontier.dataset_vs_model.compare_weighting [--size-weighted DIR] [--equal DIR]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = Path("results/cia_frontier/dataset_vs_model/results")


def settings(root: Path) -> dict:
    data = json.loads((root / "frontier.json").read_text())
    return {(s["dataset"], s["privacy"], s["noise_ratio"]): s for s in data["settings"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size-weighted", type=Path, default=BASE / "stage_c")
    parser.add_argument("--equal", type=Path, default=BASE / "stage_c_eqw")
    args = parser.parse_args()
    sw, eq = settings(args.size_weighted), settings(args.equal)
    rows = []
    for key in sorted(set(sw) | set(eq)):
        a, b = sw.get(key, {}), eq.get(key, {})
        rows.append({"dataset": key[0], "privacy": key[1], "noise_ratio": key[2],
                     **{f"{f}_{tag}": d.get(f) for tag, d in (("sw", a), ("eq", b))
                        for f in ("rounds", "accuracy", "mean_score", "min_score", "max_score")}})
    out = {"size_weighted_root": str(args.size_weighted), "equal_root": str(args.equal),
           "note": "accuracy = IN-trajectory server accuracy at round R; mean_score = mean over 7 "
                   "targets of the round-matched per-client IN/OUT score (0.5 = no signal)",
           "settings": rows}
    (args.equal / "weighting_comparison.json").write_text(json.dumps(out, indent=1) + "\n")
    fmt = lambda v, f: "-" if v is None else format(v, f)
    print(f"{'dataset':<10}{'privacy':<16}{'ratio':>8}{'acc SW(R50)':>13}{'acc EQ(R57)':>13}"
          f"{'score SW':>10}{'score EQ':>10}")
    for r in rows:
        print(f"{r['dataset']:<10}{r['privacy']:<16}{r['noise_ratio']:>8}"
              f"{fmt(r['accuracy_sw'], '.4f'):>13}{fmt(r['accuracy_eq'], '.4f'):>13}"
              f"{fmt(r['mean_score_sw'], '.3f'):>10}{fmt(r['mean_score_eq'], '.3f'):>10}")


if __name__ == "__main__":
    main()
