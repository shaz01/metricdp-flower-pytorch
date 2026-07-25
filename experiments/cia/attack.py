"""Relative-difference attack score for the paper's first-round CIA (Section 7.4.1)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CiaResult:
    privacy: str
    aggregation: str
    aggregated_test_loss: float
    target_shadow_loss: float
    difference_pct: float


def relative_difference(aggregated_loss: float, target_loss: float) -> float:
    """``(target - aggregated) / target * 100``, matching Tables 10-11.

    Verified against the paper's own numbers: aggregated=1.032, target=1.182
    gives 12.69%, matching the published 12.719% within table-rounding noise;
    dividing by ``aggregated_loss`` instead gives 14.5%, which does not match.
    """
    if target_loss == 0:
        raise ValueError("target_loss must be non-zero to compute a relative difference.")
    return (target_loss - aggregated_loss) / target_loss * 100


def make_cia_result(
    *,
    privacy: str,
    aggregation: str,
    aggregated_test_loss: float,
    target_shadow_loss: float,
) -> CiaResult:
    return CiaResult(
        privacy=privacy,
        aggregation=aggregation,
        aggregated_test_loss=aggregated_test_loss,
        target_shadow_loss=target_shadow_loss,
        difference_pct=relative_difference(aggregated_test_loss, target_shadow_loss),
    )
