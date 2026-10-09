"""Gate variants on hand-built validation scores."""

import numpy as np
import pytest

from experiments.stacked_head import gates

CE = np.array([[0.70, 0.66, 0.60, 0.75]])   # base, then three steps; the third is best on CE
ACC = np.array([[0.80, 0.82, 0.78, 0.81]])  # ... but it lowers validation accuracy


def test_v0_is_cross_entropy_argmin():
    assert gates.picks("V0", CE, ACC)[0] == 2


def test_v1_requires_accuracy_not_below_the_base_and_falls_back_to_base():
    assert gates.picks("V1", CE, ACC)[0] == 1            # steps 1 and 3 qualify; step 1 has the lower CE
    assert gates.picks("V1", CE, np.array([[0.80, 0.70, 0.70, 0.70]]))[0] == 0  # nothing qualifies but the base


def test_v2_takes_highest_accuracy_among_ce_improving_steps_and_breaks_ties_by_ce():
    assert gates.picks("V2", CE, ACC)[0] == 1            # CE-improving: 1,2 (and base is not an improvement); best accuracy is step 1
    assert gates.picks("V2", np.array([[0.70, 0.71, 0.75, 0.80]]), ACC)[0] == 0   # no CE improvement: base
    tie = gates.picks("V2", np.array([[0.70, 0.66, 0.60, 0.75]]), np.array([[0.80, 0.82, 0.82, 0.81]]))[0]
    assert tie == 2                                        # equal best accuracy -> lower CE


def test_v3_allows_a_one_point_accuracy_dip():
    assert gates.picks("V3", CE, np.array([[0.80, 0.82, 0.795, 0.81]]))[0] == 2   # 0.795 >= 0.80 - 0.01
    assert gates.picks("V3", CE, ACC)[0] == 1   # step 2 (accuracy .78) is excluded even with the slack


def test_unknown_variant_and_per_release_independence():
    with pytest.raises(ValueError):
        gates.picks("V9", CE, ACC)
    two = gates.picks("V1", np.vstack([CE, CE]), np.vstack([ACC, np.array([[0.80, 0.70, 0.70, 0.70]])]))
    assert list(two) == [1, 0]


def make_report(dips_v0, dips_v1, groups=10, reliable_v1=10):
    def block(dip_share, reliable):
        return {"groups": {f"t{i}|q0.65": {"median_gain": 0.02, "reliable": i < reliable, "n_sets": 10, "p_win": 1.0, "p_loss": 0.0, "mean_gain": 0.02, "p10_gain": 0.01, "p_win_ci95": (0.7, 1.0), "p_loss_ci95": (0.0, 0.3)} for i in range(groups)},
                "groups_reliable": reliable, "n_groups": groups, "accuracy_dip_share": dip_share, "accuracy_dip_ci95": (0.0, 0.1), "mean_accuracy_delta": 0.0, "worst_accuracy_delta": -0.01, "share_cells_at_base": 0.0}
    return {"n_cells": 100, "variants": {"V0": block(dips_v0, 10), "V1": block(dips_v1, reliable_v1), "V2": block(dips_v0, 10), "V3": block(dips_v0, 10)}}


def test_selection_adopts_only_with_a_30_percent_relative_dip_reduction():
    assert gates.select_variant(make_report(0.20, 0.10))["chosen"] == "V1"
    assert gates.select_variant(make_report(0.20, 0.15))["chosen"] == "V0"          # only 25% lower
    assert gates.select_variant(make_report(0.20, 0.05, reliable_v1=9))["chosen"] == "V0"   # not reliable everywhere


def test_confirmation_requires_reliability_and_for_non_v0_a_dip_reduction():
    good = gates.confirm(make_report(0.20, 0.10), "V1")
    assert good["confirmed"] and good["dip_relative_reduction_vs_v0"] == pytest.approx(0.5)
    assert not gates.confirm(make_report(0.20, 0.18), "V1")["confirmed"]
    assert not gates.confirm(make_report(0.20, 0.05, reliable_v1=8), "V1")["confirmed"]
    assert gates.confirm(make_report(0.20, 0.30), "V0")["confirmed"]
