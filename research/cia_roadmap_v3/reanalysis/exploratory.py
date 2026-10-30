"""Exploratory follow-ups to reanalyze.py, NOT pre-registered (decided after reading its results, 2026-10-30).

1. Does logit adjustment add privacy once the schedule is known? C1 against B1 q = 0.25, both under the
   sampling-aware attacker, at A1's targets.
2. The strongest attacker available on C1: defense-aware (logit-adjusted) shadow losses plus the schedule statistic,
   against metric privacy under the standard attacker.

    PYTHONPATH="$PWD" uv run --no-sync python research/cia_roadmap_v3/reanalysis/exploratory.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("reanalyze", HERE / "reanalyze.py")
rz = importlib.util.module_from_spec(_spec)
sys.modules["reanalyze"] = rz
_spec.loader.exec_module(rz)


def combined(config):
    """Logit-adjusted shadow losses (C1's defense-aware attacker) plus the centred schedule statistic on them."""
    a = config.audit
    z = rz.schedule_statistic(a.adjusted_losses, rz.simulated_schedules(config), config.sampling_rate)
    features = np.concatenate([a.adjusted_losses, (z / z.std())[..., None]], axis=2)
    return rz.Scored(rz.calibrated_scores(features, a.membership), a.membership, a.final_accuracy)


def main():
    c1 = [rz.load_config(rz.ROOTS["c1"], n) for n in rz.C1]
    b25 = [rz.load_config(rz.ROOTS["b1"], n) for n in rz.B1_Q25]
    mp = [rz.standard(rz.load_config(rz.ROOTS["a1"], n)) for n in rz.MP]
    sa_c1, sa_b25 = [rz.sampling_aware(c) for c in c1], [rz.sampling_aware(c) for c in b25]
    comb = [combined(c) for c in c1]
    out = {
        "note": "exploratory, not pre-registered",
        "c1_sa_vs_b1q25_sa": [rz.matched(sa_c1, sa_b25, t) for t in rz.TARGETS],
        "c1_combined_vs_mp": [rz.matched(comb, mp, t) for t in rz.TARGETS],
        "c1_combined_curve": [(float(p.accuracy.mean()), rz.roc_auc(p.scores.ravel(), p.membership.ravel()))
                              for p in comb],
    }
    (HERE / "exploratory.json").write_text(json.dumps(out, indent=1, default=float))
    for key in ("c1_sa_vs_b1q25_sa", "c1_combined_vs_mp"):
        print(key, [(r["target"], round(r["gate"]["estimate"], 3), round(r["gate"]["low"], 3),
                     round(r["gate"]["high"], 3), round(r["two_way"]["low"], 3), round(r["two_way"]["high"], 3))
                    for r in out[key]])
    print("combined curve", [(round(a, 3), round(u, 3)) for a, u in out["c1_combined_curve"]])


if __name__ == "__main__":
    main()
