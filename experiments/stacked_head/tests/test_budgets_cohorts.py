"""ANOVA components and the budget-map rule on synthetic data with known structure."""

import json

import numpy as np
import pytest

from experiments.stacked_head import budgets_cohorts as bc


def test_anova_recovers_known_variance_components():
    rng = np.random.default_rng(0)
    sets = rng.normal(0, 0.03, size=(2000, 1))       # many replicates of the 10x4 design, averaged over replicates below
    comps = []
    for _ in range(300):
        table = rng.normal(0, 0.03, size=(10, 1)) + rng.normal(0, 0.005, size=(1, 4)) + rng.normal(0, 0.01, size=(10, 4))
        comps.append(bc.anova_components(table))
    assert np.mean([c["sigma_set"] for c in comps]) == pytest.approx(0.03**2, rel=0.25)
    assert np.mean([c["sigma_cohort"] for c in comps]) == pytest.approx(0.005**2, abs=2e-5)
    assert np.mean([c["sigma_resid"] for c in comps]) == pytest.approx(0.01**2, rel=0.25)


def test_anova_with_no_cohort_effect_has_a_near_zero_cohort_share_and_shares_sum_to_one():
    rng = np.random.default_rng(1)
    table = rng.normal(0, 0.05, size=(10, 1)) + rng.normal(0, 0.002, size=(10, 4))
    c = bc.anova_components(table)
    assert c["share_set"] + c["share_cohort"] + c["share_resid"] == pytest.approx(1.0)
    assert c["cohort_to_set_ratio"] < 0.1


def write(directory, task, risk, budget, public_set, cohort, tag, gain):
    name = f"{task}_b{budget}_s{public_set}_{cohort}_q{int(round(risk * 100))}_{tag}"
    result = {"run_name": name, "config": {"task": task, "risk": risk, "public-set": public_set, "budget": budget, "cohort": cohort},
              "control": {"ce": 0.6, "accuracy": 0.8}, "summary": {"gain_over_control": gain, "accuracy_delta": 0.0, "pick_fraction": {}}}
    (directory / f"{name}.json").write_text(json.dumps(result))


def test_budget_map_reliability_and_fade_budget(tmp_path):
    tasks = [f"t{i}" for i in range(5)]
    # 32: all groups win; 128: half the sets win; 512: no wins. Needs 10 groups = 5 tasks x 2 risks.
    for task in tasks:
        for risk in (0.65, 0.8):
            for s in range(10):
                write(tmp_path, task, risk, 32, s, "A", "sweep", 0.02)
                write(tmp_path, task, risk, 128, s, "A", "budgetmap", 0.01 if s < 5 else -0.001)
                write(tmp_path, task, risk, 512, s, "A", "budgetmap", -0.001)
    report = bc.budget_map(tmp_path)
    assert report["budgets"]["32"]["reliable"] and report["budgets"]["32"]["groups_reliable"] == 10
    assert not report["budgets"]["128"]["reliable"] and report["budgets"]["128"]["pooled_p_win"] == pytest.approx(0.5)
    assert report["fade_budget"] == 512 and "fade budget" in bc.format_budget_map(report)


def test_cohort_report_claim_depends_on_the_variance_ratio(tmp_path):
    rng = np.random.default_rng(2)
    for task in ("kmnist_classes0to3", "kmnist_classes4to7"):
        for risk in (0.65, 0.8):
            effect = rng.normal(0, 0.03, size=10)
            for s in range(10):
                for cohort, tag in (("A", "sweep"), ("B", "cohorts"), ("C", "cohorts"), ("D", "cohorts")):
                    write(tmp_path, task, risk, 32, s, cohort, tag, 0.03 + effect[s] + rng.normal(0, 0.002))
    report = bc.cohort_report(tmp_path)
    assert len(report["groups"]) == 4 and report["claim_cohort_small_vs_set"]
    assert "claim" in bc.format_cohorts(report)
