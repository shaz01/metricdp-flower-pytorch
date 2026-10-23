"""M2 analysis (synthetic; no training)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.cia.inclusion import balanced_design
from results.cia_v2.frontier import analyze
from results.cia_v2.randomized_inclusion.analysis import Audit
from results.cia_v2.randomized_inclusion.runner import CANDIDATES


def test_formal_mu_for_equal_weights():
    assert analyze.formal_mu(0.001546) == pytest.approx(math.sqrt(50) / (32 * 0.001546))
    assert analyze.formal_mu(0.0) == math.inf


@pytest.mark.parametrize("accuracies,expected", [
    ({0.001546: 0.80, 0.006184: 0.78, 0.024736: 0.70}, ["8x"]),
    ({0.001546: 0.80, 0.006184: 0.70, 0.024736: 0.68}, ["2x"]),
    ({0.001546: 0.80, 0.006184: 0.79, 0.024736: 0.77}, []),
])
def test_fill_in_rule_uses_accuracy_only(accuracies, expected):
    assert analyze.fill_in_levels(accuracies) == expected


def summary(ratio, upper, top_point, accuracy_low):
    return {"noise_ratio": ratio, "all": {"auc": {"high": upper}}, "top-decile": {"auc": {"estimate": top_point}},
            "accuracy": {"low": accuracy_low}}


def test_gate_finds_the_lowest_useful_noise():
    summaries = [summary(0.001546, 0.75, 0.80, 0.78), summary(0.006184, 0.58, 0.59, 0.74),
                 summary(0.024736, 0.52, 0.50, 0.65)]
    gate = analyze.g_m2(summaries)
    assert gate["verdict"] == "useful empirical regime"
    assert gate["r_0.6"] == 0.006184
    assert gate["operating_points"] == [0.006184, 0.001546]


def test_gate_needs_the_top_decile_point_and_accuracy():
    summaries = [summary(0.001546, 0.58, 0.65, 0.78), summary(0.006184, 0.55, 0.55, 0.69)]
    assert analyze.g_m2(summaries)["verdict"] == "no useful regime"


def synthetic_audit(shift, replicates=32, seed=0):
    rng = np.random.default_rng(seed)
    membership = balanced_design(len(CANDIDATES), replicates, seed=seed)
    losses = rng.normal(size=(len(CANDIDATES), replicates, 11)) - shift * membership[..., None]
    return Audit(config="x", rounds=tuple(range(0, 51, 5)), losses=losses, membership=membership,
                 final_accuracy=rng.uniform(0.7, 0.8, replicates), released_norms=None, surrogate_losses=None,
                 run_names=tuple(map(str, range(replicates))), min_client_accuracy=rng.uniform(0.3, 0.5, replicates))


def test_config_summary_reports_gate_and_reference_intervals():
    result = analyze.config_summary(synthetic_audit(0.5), noise_ratio=0.001546, resamples=100)
    for subset in ("all", "top-decile"):
        auc = result[subset]["auc"]
        assert auc["low"] <= auc["estimate"] <= auc["high"]
        assert result[subset]["auc_two_way"]["low"] <= auc["estimate"]
        assert {"tpr_at_5pct_fpr", "tpr_at_1pct_fpr", "auc_final_round", "auc_threshold"} <= set(result[subset])
    assert result["all"]["auc"]["estimate"] > 0.6
    assert 0.7 <= result["accuracy"]["low"] <= result["accuracy"]["estimate"] <= result["accuracy"]["high"] <= 0.8
    assert result["formal_mu"] == pytest.approx(analyze.formal_mu(0.001546))


def test_vanilla_summary_is_valid_json():
    import json

    result = analyze.config_summary(synthetic_audit(0.5), noise_ratio=0.0, resamples=50)
    assert result["formal_mu"] is None
    json.dumps(result, allow_nan=False)


def test_surrogate_auc_is_reported_when_surrogates_were_scored():
    from dataclasses import replace

    audit = synthetic_audit(0.5)
    with_surrogates = replace(audit, surrogate_losses=audit.losses + 0.01)
    result = analyze.config_summary(with_surrogates, noise_ratio=0.001546, resamples=50)
    assert result["all"]["auc_surrogate"] == pytest.approx(result["all"]["auc"]["estimate"], abs=0.05)
    assert analyze.config_summary(audit, noise_ratio=0.001546, resamples=50)["all"]["auc_surrogate"] is None


def test_formal_mu_uses_the_audit_rounds():
    from dataclasses import replace

    audit = replace(synthetic_audit(0.5), rounds=tuple(range(0, 101, 10)))
    result = analyze.config_summary(audit, noise_ratio=0.003, resamples=20)
    assert result["formal_mu"] == pytest.approx(math.sqrt(100) / (32 * 0.003))
