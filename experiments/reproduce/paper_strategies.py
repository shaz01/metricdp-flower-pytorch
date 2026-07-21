r"""
Strategy hyperparameters from Table 4:
 - FedAvg and FedMedian take nothing
 - FedAvgM $\beta=0.5, \mu=0.1$
 - FedProx $\mu=0.5$; FedOpt $\beta_1=\beta_2=0, \tau=10^{-9}$
 - FedYogi $\beta_1=0.9, \beta_2=0.99, \tau=10^{-3}$

The stateful strategies (FedAvgM, FedOpt, FedYogi) need initial global
parameters: 20 epochs, batch 32, on the stratified validation half. DP
parameters: clipping norm $C=5$, noise multiplier $n_\epsilon=0.01$, and for
metric-private runs the per-round distance divides the noise multiplier.
"""

from __future__ import annotations

from collections.abc import Iterable

from flwr.serverapp.strategy import Strategy

from metricdp_pytorch.strategy_factory import (
    AGGREGATION_METHODS,
    PRIVACY_MODES,
    make_strategy,
)

PAPER_NUM_CLIENTS = 4
PAPER_CLIPPING_NORM = 5.0
PAPER_NOISE_MULTIPLIER = 0.01


def create_paper_strategy(
    *,
    aggregation: str,
    privacy: str,
    num_clients: int = PAPER_NUM_CLIENTS,
    fraction_evaluate: float = 1.0,
    noise_multiplier: float = PAPER_NOISE_MULTIPLIER,
    clipping_norm: float = PAPER_CLIPPING_NORM,
) -> Strategy:
    return make_strategy(
        aggregation=aggregation,
        privacy=privacy,
        num_clients=num_clients,
        fraction_evaluate=fraction_evaluate,
        noise_multiplier=noise_multiplier,
        clipping_norm=clipping_norm,
    )


def create_paper_strategies(
    *,
    num_clients: int = PAPER_NUM_CLIENTS,
    fraction_evaluate: float = 1.0,
    aggregations: Iterable[str] = AGGREGATION_METHODS,
    privacy_modes: Iterable[str] = PRIVACY_MODES,
    noise_multiplier: float = PAPER_NOISE_MULTIPLIER,
    clipping_norm: float = PAPER_CLIPPING_NORM,
) -> dict[tuple[str, str], Strategy]:
    """Create the paper's privacy × aggregation strategy matrix.

    By default this returns 18 independent Flower strategies, keyed as
    ``(privacy_mode, aggregation)``. It delegates all algorithm-specific
    parameters to :func:`metricdp_pytorch.strategy_factory.make_strategy`.
    """
    selected_aggregations = tuple(aggregations)
    selected_privacy_modes = tuple(privacy_modes)
    return {
        (privacy, aggregation): create_paper_strategy(
            aggregation=aggregation,
            privacy=privacy,
            num_clients=num_clients,
            fraction_evaluate=fraction_evaluate,
            noise_multiplier=noise_multiplier,
            clipping_norm=clipping_norm,
        )
        for privacy in selected_privacy_modes
        for aggregation in selected_aggregations
    }
