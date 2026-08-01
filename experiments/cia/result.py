"""Client-inference attack result records and scoring."""

from __future__ import annotations

from dataclasses import dataclass

from experiments.reproduce.matrix import Combo


@dataclass(frozen=True)
class CiaResult:
    partition_mode: str
    num_clients: int
    privacy: str
    aggregation: str
    noise_multiplier: float
    server_round: int
    aggregated_test_loss: float
    target_shadow_loss: float
    shadow_fraction: float
    shadow_size: int
    difference_pct: float


def relative_difference(aggregated_loss: float, target_loss: float) -> float:
    """``(target - aggregated) / target * 100``, matching Tables 10-11."""
    if target_loss == 0:
        raise ValueError("target_loss must be non-zero to compute a relative difference.")
    return (target_loss - aggregated_loss) / target_loss * 100


def make_cia_result(
    *,
    combo: Combo,
    server_round: int,
    aggregated_test_loss: float,
    target_shadow_loss: float,
    shadow_fraction: float,
    shadow_size: int,
) -> CiaResult:
    return CiaResult(
        partition_mode=combo.partition,
        num_clients=combo.num_clients,
        privacy=combo.privacy,
        aggregation=combo.aggregation,
        noise_multiplier=combo.noise_multiplier,
        server_round=server_round,
        aggregated_test_loss=aggregated_test_loss,
        target_shadow_loss=target_shadow_loss,
        shadow_fraction=shadow_fraction,
        shadow_size=shadow_size,
        difference_pct=relative_difference(aggregated_test_loss, target_shadow_loss),
    )
