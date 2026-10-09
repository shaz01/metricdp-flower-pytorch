"""Consistency audit of finished cells: every summary must be recomputable from the stored per-release arrays.

For each ``<name>.json`` with a ``<name>.releases.npz`` the audit recomputes, from the stored validation and held-out
CE of all step multipliers, the gate picks, the gated/alternative/fixed gains and the per-round records, and checks
them against the result file. It also checks the control against the bundle metadata and that no round is missing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit_cell(path: Path) -> int:
    result = json.loads(path.read_text(encoding="utf-8"))
    arrays = np.load(path.with_suffix(".releases.npz"))
    multipliers = [float(m) for m in arrays["multipliers"]]
    eval_ce, eval_accuracy, val_ce = arrays["eval_ce"].astype(float), arrays["eval_accuracy"].astype(float), arrays["val_ce"].astype(float)
    rounds = result["rounds"]
    count = len(rounds)
    assert eval_ce.shape == (count, len(multipliers)) == val_ce.shape == eval_accuracy.shape, path.name
    assert [r["round"] for r in rounds] == list(range(1, count + 1)), path.name
    control = result["control"]
    assert abs(control["ce"] - result["bundle"]["control"]["ce"]) < 1e-9 and abs(arrays["control"][0] - control["ce"]) < 1e-6
    rows = np.arange(count)
    picks = val_ce.argmin(axis=1)
    assert [multipliers[i] for i in picks] == [r["multiplier"] for r in rounds], f"{path.name}: strategy pick != recomputed pick"
    gated = eval_ce[rows, picks]
    assert np.allclose(gated, [r["eval-ce"] for r in rounds], atol=2e-5), path.name
    summary = result["summary"]
    assert abs(summary["gain_over_control"] - (control["ce"] - gated.mean())) < 2e-5, path.name
    assert abs(summary["accuracy_delta"] - (eval_accuracy[rows, picks].mean() - control["accuracy"])) < 2e-5, path.name
    checks = 6
    for size, block in (result["summary_alt_validation"] or {}).items():
        alt_picks = arrays[f"val{size}_ce"].astype(float).argmin(axis=1)
        assert abs(block["gain_over_control"] - (control["ce"] - eval_ce[rows, alt_picks].mean())) < 2e-5, path.name
        checks += 1
    if result["summary_fixed_multiplier_1"] is not None:
        column = multipliers.index(1.0)
        assert abs(result["summary_fixed_multiplier_1"]["gain_over_control"] - (control["ce"] - eval_ce[:, column].mean())) < 2e-5, path.name
        checks += 1
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, nargs="?", default=Path("results/stacked_head"))
    parser.add_argument("--tags", nargs="+", default=["sweep", "transfer"])
    args = parser.parse_args()
    cells, checks = 0, 0
    for tag in args.tags:
        for path in sorted(args.directory.glob(f"*_{tag}.json")):
            if path.name.startswith("sweep_report_"):
                continue  # derived report, not a run result
            checks += audit_cell(path)
            cells += 1
    print(f"release audit passed: {cells} cells, {checks} checks")


if __name__ == "__main__":
    main()
