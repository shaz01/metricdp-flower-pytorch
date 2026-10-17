"""Audit design, seeds, bounds and analysis on synthetic records (no training, no dataset)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.frozen_head import audit


def test_design_is_balanced_with_fixed_count():
    design = audit.design()
    assert design.shape == (32, 32)
    assert np.all(design.sum(axis=1) == 16) and np.all(design.sum(axis=0) == 16)


def test_replicate_seeds_independent_and_deterministic():
    seeds = [audit.replicate_seeds(j) for j in range(audit.AUDIT_REPLICATES)]
    assert len({s for pair in seeds for s in pair}) == 2 * audit.AUDIT_REPLICATES
    assert seeds[3] == audit.replicate_seeds(3)


def test_gdp_bounds():
    bounds = audit.gdp_bounds(1.0)
    assert bounds["auc"] == pytest.approx(0.76025, abs=1e-5)
    assert bounds["tpr_at_5pct"] == pytest.approx(0.259511, abs=1e-5)


def _records(shift: float, rounds: int = 5, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    design = audit.design()
    difficulty = rng.normal(0, 1, (32, 1))
    records = []
    for j in range(audit.AUDIT_REPLICATES):
        member = design[:, j]
        losses = difficulty + rng.normal(0, 1, (32, rounds)) - shift / math.sqrt(rounds) * member[:, None]
        records.append({"job": {"clients": 48, "rounds": rounds, "mu_total": 1.0, "replicate": j},
                        "membership": member.astype(int).tolist(), "shadow_losses": losses.tolist(),
                        "candidates": list(range(32)), "top5": [0, 1, 2, 3, 4], "final_server_test": 0.9})
    return records


def test_analysis_recovers_no_leakage_and_strong_leakage():
    none = audit.analyze(_records(0.0))["settings"][0]
    strong = audit.analyze(_records(3.0, seed=1))["settings"][0]
    low, high = none["calibrated_all_auc"]["jackknife_975"]
    assert low < 0.5 < high
    assert strong["calibrated_all_auc"]["estimate"] > 0.9
    assert strong["bound_violated"]  # a mu = 3 signal exceeds the mu = 1 bound
    assert not none["bound_violated"]


def test_public_size_audit_plan_is_the_mu_at_most_2_settings_in_its_own_directory():
    jobs = audit.audit_jobs(20)
    assert len(jobs) == 12 * audit.AUDIT_REPLICATES
    assert {job.mu_total for job in jobs} == {0.0, 1.0, 2.0}
    assert all(job.path.parent.name == "audit_public20" and job.name.endswith(f"_pub20_r{job.replicate:02d}")
               for job in jobs)
    assert audit.AuditJob(48, 10, 1.0, 3).name == "n48_T10_mu1_r03"  # Part C names unchanged
    assert len(audit.audit_jobs()) == 640
