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

Deliberate deviations from the literal Table 4 values, made after
reproducing catastrophic training collapse in real 8-client runs and
confirming the mechanism with controlled experiments (see
``metricdp_pytorch.strategy_factory.make_base_strategy`` for the detailed
reasoning, including why the paper's own officially published code and
published results don't resolve the discrepancy):

- FedOpt uses $\tau=10^{-3}$ rather than $10^{-9}$ (verified irrelevant in
  practice) and an explicit, decaying $\eta$ (starting at $\eta_0=0.01$,
  see ``metricdp_pytorch.strategy_factory.DecayingEtaFedAdam``) rather
  than Flower's default constant $\eta=0.1$ (Table 4 doesn't list $\eta$
  at all). With $\beta_1=\beta_2=0$, FedAdam's update degenerates into a
  fixed $\pm\eta$ step per parameter regardless of true gradient size;
  $\eta=0.1$ is ~10-20x this model's real weight scale in its dominant FC
  layer and destroys it within one round, confirmed independent of
  PyTorch init/pretraining quality. A constant $\eta=0.01$ avoids the
  collapse but visibly oscillates late in training (zero momentum means
  no round-to-round smoothing); decaying $\eta$ resolves that too,
  converging smoothly to 0.9625 accuracy over 20 real federated rounds.
- FedProx keeps the literal $\mu=0.5$, but the proximal term itself is
  normalized by total parameter count (mean squared difference, matching
  TensorFlow/Keras's idiomatic ``reduce_mean`` -- the paper's actual
  framework) rather than an unnormalized sum. Real federated testing
  showed $\mu=0.5$ traps training at an identical, frozen majority-class
  accuracy every round under both the paper's squared-full-vector-norm
  formula and Flower's own unsquared per-tensor-norm convention -- both
  unnormalized, both dominated by this model's one 100352->64 FC layer.
  Normalizing by parameter count resolves it: $\mu=0.5$ then converges at
  essentially the same rate as unregularized FedAvg, matching the paper's
  own Table 6 result that FedProx and FedAvg land at identical accuracy
  (0.909).

All of the literal Table 4 formulas/values are mathematically faithful to
the paper but numerically destructive for this model's ~6.5M parameters
(98.6% in one 100352->64 FC layer).
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
