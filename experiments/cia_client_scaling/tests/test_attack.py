"""Tests for the 48-client CIA attack-score record."""

from __future__ import annotations

import pytest

from experiments.cia_client_scaling.attack import CiaScalingResult, make_cia_scaling_result


def test_make_cia_scaling_result_computes_difference_pct() -> None:
    result = make_cia_scaling_result(
        partition_mode="homogeneous",
        timing="first-round",
        privacy="vanilla",
        aggregation="fedavg",
        aggregated_test_loss=1.032,
        target_shadow_loss=1.182,
        shadow_size=8,
    )
    assert isinstance(result, CiaScalingResult)
    assert result.partition_mode == "homogeneous"
    assert result.timing == "first-round"
    assert result.privacy == "vanilla"
    assert result.aggregation == "fedavg"
    assert result.shadow_size == 8
    assert result.difference_pct == pytest.approx(12.719, abs=0.1)


def test_make_cia_scaling_result_tags_post_convergence_timing() -> None:
    result = make_cia_scaling_result(
        partition_mode="non-iid",
        timing="post-convergence",
        privacy="metric-privacy",
        aggregation="fedyogi",
        aggregated_test_loss=0.8,
        target_shadow_loss=1.0,
        shadow_size=11,
    )
    assert result.timing == "post-convergence"
    assert result.shadow_size == 11
    assert result.difference_pct == pytest.approx(20.0)
