"""Compare a probe-noise run through the real Flower message path with the stored research-probe cell.

The CNN-head research probe (``results/client_specific_noise/cnn_head.{json,npz}``) saved the per-release gated
held-out CE of every cell. A run with ``--noise-source probe`` replays the same noise draws through the ServerApp and
ClientApp code, so its per-round ``eval-ce`` must match up to float32/float64 round-off in the model arrays.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COHORTS = ("A", "B")


def compare(result_path: Path, probe_json: Path, probe_npz: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = result["config"]
    task, budget, subset, cohort, risk = config["task"], int(config["budget"]), int(config["public-set"]), config["cohort"], float(config["risk"])
    if config.get("noise-source") != "probe":
        raise ValueError("Equivalence needs a run with noise-source=probe.")
    key = f"{task}_b{budget}_s{subset}_{cohort}_noisy_{int(round(risk * 100))}_gated_ce"
    stored = np.load(probe_npz)[key]
    rounds = len(result["rounds"])
    ours = np.array([row["eval-ce"] for row in sorted(result["rounds"], key=lambda r: r["round"])])
    difference = np.abs(ours - stored[:rounds])
    probe = json.loads(probe_json.read_text(encoding="utf-8"))
    cell = next(c for c in probe["tasks"][task] if c["budget"] == budget and c["subset"] == subset and c["cohort"] == cohort)
    probe_control = cell["control"]["ce"]
    return {
        "cell": key, "releases_compared": rounds, "max_abs_eval_ce_difference": float(difference.max()), "mean_abs_eval_ce_difference": float(difference.mean()),
        "control_ce": {"flower_path": result["control"]["ce"], "probe": probe_control, "abs_difference": abs(result["control"]["ce"] - probe_control)},
        "mean_gated_ce": {"flower_path": float(ours.mean()), "probe_same_rounds": float(stored[:rounds].mean())},
        "base_selection": {"flower_path": result["bundle"]["base"], "probe": cell["base"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--probe-json", type=Path, default=Path("results/client_specific_noise/cnn_head.json"))
    parser.add_argument("--probe-npz", type=Path, default=Path("results/client_specific_noise/cnn_head.npz"))
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()
    report = compare(args.result, args.probe_json, args.probe_npz)
    print(json.dumps(report, indent=1))
    ok = report["max_abs_eval_ce_difference"] <= args.tolerance and report["control_ce"]["abs_difference"] <= args.tolerance
    print("EQUIVALENT" if ok else "NOT EQUIVALENT", f"(tolerance {args.tolerance})")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
