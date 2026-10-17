"""M1 analysis: v1's paired clean-view score on the decoupled trajectories, and the G-M1 verdict.

The statistic, its pooling and its cluster bootstrap are v1's (``results/cia_frontier/signature_suppression/analyze.py``:
``load``, ``cell_statistics``; 2,000 resamples over the five target clusters, seed 1729 as in step 3), so only the noise
coupling differs from ``results/cia_frontier/consensus_kl/analysis_step3.json``. The paired score is an effect size
(measurement standard v2, item 6), not attack success. Also reported: the realized-noise-energy correlation between
the IN and each OUT trajectory (``research/cia_roadmap_v2/scripts/measurement_audit.py`` statistic, per round), which
should be near 0 here and was 0.93-1.00 in v1.

    uv run --no-sync python -m results.cia_v2.measurement.m1.analyze [root] --out results/cia_v2/measurement/m1/analysis.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

from results.cia_frontier.signature_suppression.analyze import cell_statistics, load

HERE = Path(__file__).resolve().parent
COUPLED = {"gdp-r0.001546": 0.788, "gdp-r0.006184": 0.792}  # v1 analysis_step3.json, paired, shared noise
TOP_DECILE = (23, 7)
RESAMPLES = 2000
BOOTSTRAP_SEED = 1729
GATE_CELL = "gdp-r0.006184"


def verdict(score_4x):
    """G-M1, fixed in ROADMAP.md before any run."""
    if score_4x <= 0.70:
        return "confirmed"
    if score_4x < 0.75:
        return "partial"
    return "not material"


def noise_residual(folder):
    """Per round ``||Delta||^2 - ||s||^2 - E||xi||^2`` (the measurement-audit statistic)."""
    (training,) = [p for p in glob.glob(f"{folder}/*.json") if re.search(r"__eurosat_cnn\.json$", p)]
    metrics = json.load(open(training))["train_metrics"]
    releases = {r["round"]: r["update_norm"] for r in json.load(open(f"{folder}/releases.json"))}
    rounds = sorted(map(int, metrics))
    return np.array([releases[t] ** 2 - metrics[str(t)]["dp-signal-update-norm"] ** 2
                     - metrics[str(t)]["dp-expected-noise-l2-norm"] ** 2 for t in rounds])


def coupling(cell_root):
    worlds = {}
    for manifest in glob.glob(f"{cell_root}/*/manifest.json"):
        folder = os.path.dirname(manifest)
        worlds[json.load(open(manifest))["out_target"]] = noise_residual(folder)
    return {str(target): float(np.corrcoef(worlds[None], series)[0, 1])
            for target, series in worlds.items() if target is not None}


def evaluate(root, resamples=RESAMPLES):
    data = load(root)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    report = {}
    for cell, values in sorted(data.items()):
        clusters = sorted(values["cells"])
        gaps = [values["cells"][k]["clean"] for k in clusters]
        point = cell_statistics(gaps)
        draws = [rng.integers(0, len(clusters), len(clusters)) for _ in range(resamples)]
        boot = [cell_statistics([gaps[i] for i in d]) for d in draws]
        top = cell_statistics([values["cells"][k]["clean"] for k in clusters if k[1] in TOP_DECILE])
        report[cell] = dict(
            paired_score=point["score"], paired_score_ci95=np.quantile([b["score"] for b in boot], [0.025, 0.975]).tolist(),
            oracle_midpoint_auc=point["auc"], top_decile_paired_score=top["score"],
            per_target_score={str(t): float((values["cells"][(s, t)]["clean"] > 0).mean()) for s, t in clusters},
            coupled_reference_score=COUPLED.get(cell), accuracy=values["accuracy"],
            noise_energy_correlation_in_vs_out=coupling(Path(root) / cell),
        )
    gate = report[GATE_CELL]["paired_score"]
    return dict(label="paired clean-view score: effect size with independent noise and client seeds per world",
                clusters="targets 23, 7, 1, 0, 3 (one seed)", resamples=resamples, bootstrap_seed=BOOTSTRAP_SEED,
                cells=report, gate={"statistic": "4x decoupled paired score", "value": gate, "verdict": verdict(gate)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=HERE / "results")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = json.dumps(evaluate(args.root), indent=2, allow_nan=False)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
