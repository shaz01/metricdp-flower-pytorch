"""Attack-score record for the 48-client CIA experiment.

Reuses ``experiments.cia.attack.relative_difference`` unmodified; only adds
``partition_mode``/``timing`` fields the 3-client CIA experiment didn't need.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.cia.attack import relative_difference


@dataclass(frozen=True)
class CiaScalingResult:
    partition_mode: str
    timing: str
    privacy: str
    aggregation: str
    aggregated_test_loss: float
    target_shadow_loss: float
    difference_pct: float


def make_cia_scaling_result(
    *,
    partition_mode: str,
    timing: str,
    privacy: str,
    aggregation: str,
    aggregated_test_loss: float,
    target_shadow_loss: float,
) -> CiaScalingResult:
    return CiaScalingResult(
        partition_mode=partition_mode,
        timing=timing,
        privacy=privacy,
        aggregation=aggregation,
        aggregated_test_loss=aggregated_test_loss,
        target_shadow_loss=target_shadow_loss,
        difference_pct=relative_difference(aggregated_test_loss, target_shadow_loss),
    )
