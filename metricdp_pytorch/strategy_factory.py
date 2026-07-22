""" Factory for the experiment matrix's aggregation and privacy strategies. """

from __future__ import annotations

from flwr.serverapp.strategy import (
    DifferentialPrivacyServerSideFixedClipping,
    FedAdam,
    FedAvg,
    FedAvgM,
    FedMedian,
    FedProx,
    FedYogi,
    Strategy,
)

from metricdp_pytorch.metricdp_strategy import (
    MetricPrivacyServerSideFixedClipping,
)

AGGREGATION_METHODS = (
    "fedavg",
    "fedavgm",
    "fedmedian",
    "fedprox",
    "fedopt",
    "fedyogi",
)
PRIVACY_MODES = ("vanilla", "global-dp", "metric-privacy")


def make_base_strategy(
    aggregation: str,
    *,
    num_clients: int,
    fraction_evaluate: float = 1.0,
) -> Strategy:
    """Construct one of the six paper aggregation strategies.

    ``fedopt`` is instantiated as FedAdam with the paper's beta1=beta2=0
    configuration, but tau=1e-3 rather than Table 4's literal tau=1e-9. With
    beta_1=beta_2=0, FedAdam's update reduces to ``eta * delta_t / (|delta_t|
    + tau)``: for any per-parameter delta much larger than tau (verified true
    of tau=1e-9 AND tau=1e-7 for this model's realistic delta magnitudes,
    ~1e-4 to 1e-2), this normalizes to a fixed +-eta step regardless of the
    true signal's size, destroying an already-good pretrained model within
    one round. tau=1e-3 is large enough relative to those deltas to restore
    genuine adaptive scaling; it's also FedYogi's tau, which converges
    cleanly with the same beta values pattern. FedYogi and FedAvgM use the
    values listed in the paper's experimental setup as-is.
    """
    common = {
        "fraction_evaluate": fraction_evaluate,
        "min_train_nodes": num_clients,
        "min_evaluate_nodes": num_clients,
        "min_available_nodes": num_clients,
    }
    if aggregation == "fedavg":
        return FedAvg(**common)
    if aggregation == "fedavgm":
        return FedAvgM(
            **common,
            server_learning_rate=0.1,
            server_momentum=0.5,
        )
    if aggregation == "fedmedian":
        return FedMedian(**common)
    if aggregation == "fedprox":
        return FedProx(**common, proximal_mu=0.5)
    if aggregation == "fedopt":
        return FedAdam(**common, beta_1=0.0, beta_2=0.0, tau=1e-3)
    if aggregation == "fedyogi":
        return FedYogi(**common, beta_1=0.9, beta_2=0.99, tau=1e-3)
    raise ValueError(
        f"Unknown aggregation method {aggregation!r}; choose from {AGGREGATION_METHODS}."
    )


def make_strategy(
    aggregation: str,
    privacy: str,
    *,
    num_clients: int,
    fraction_evaluate: float,
    noise_multiplier: float,
    clipping_norm: float,
) -> Strategy:
    """Construct an aggregation strategy and apply the selected DP wrapper."""
    strategy = make_base_strategy(
        aggregation,
        num_clients=num_clients,
        fraction_evaluate=fraction_evaluate,
    )
    if privacy == "vanilla":
        return strategy
    if privacy == "global-dp":
        return DifferentialPrivacyServerSideFixedClipping(
            strategy=strategy,
            noise_multiplier=noise_multiplier,
            clipping_norm=clipping_norm,
            num_sampled_clients=num_clients,
        )
    if privacy == "metric-privacy":
        return MetricPrivacyServerSideFixedClipping(
            strategy=strategy,
            noise_multiplier=noise_multiplier,
            clipping_norm=clipping_norm,
            num_sampled_clients=num_clients,
        )
    raise ValueError(f"Unknown privacy mode {privacy!r}; choose from {PRIVACY_MODES}.")
