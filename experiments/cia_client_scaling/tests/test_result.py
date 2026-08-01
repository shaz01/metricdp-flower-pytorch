"""Tests for the 48-client CIA attack-score record."""

from __future__ import annotations

import pytest

from experiments.cia_client_scaling.result import make_cia_scaling_result


def test_make_cia_scaling_result_computes_difference_pct() -> None:
    result = make_cia_scaling_result(
        partition_mode="homogeneous",
        timing="first-round",
        privacy="vanilla",
        aggregation="fedavg",
        noise_multiplier=0.01,
        aggregated_test_loss=1.032,
        target_shadow_loss=1.182,
        shadow_size=8,
    )
    assert result.difference_pct == pytest.approx(12.69, abs=0.1)
