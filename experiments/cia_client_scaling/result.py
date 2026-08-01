"""Attack-score record for the 48-client CIA experiment.

Reuses ``experiments.cia.result.relative_difference`` unmodified; only adds
the ``partition_mode``/``timing``/``noise_multiplier`` fields the 3-client CIA
experiment didn't need.
"""

from __future__ import annotations

from dataclasses import dataclass

from experiments.cia.result import relative_difference


@dataclass(frozen=True)
class CiaScalingResult:
    partition_mode: str
    timing: str
    privacy: str
    aggregation: str
    noise_multiplier: float
    aggregated_test_loss: float
    target_shadow_loss: float
    shadow_size: int
    difference_pct: float


def make_cia_scaling_result(
    *,
    partition_mode: str,
    timing: str,
    privacy: str,
    aggregation: str,
    noise_multiplier: float,
    aggregated_test_loss: float,
    target_shadow_loss: float,
    shadow_size: int,
) -> CiaScalingResult:
    return CiaScalingResult(
        partition_mode=partition_mode,
        timing=timing,
        privacy=privacy,
        aggregation=aggregation,
        noise_multiplier=noise_multiplier,
        aggregated_test_loss=aggregated_test_loss,
        target_shadow_loss=target_shadow_loss,
        shadow_size=shadow_size,
        difference_pct=relative_difference(aggregated_test_loss, target_shadow_loss),
    )
