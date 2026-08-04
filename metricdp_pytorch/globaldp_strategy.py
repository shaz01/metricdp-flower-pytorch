"""Global-DP Flower wrapper with persisted per-round diagnostics."""

from __future__ import annotations

from collections.abc import Iterable

from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, Strategy
from flwr.supercore.differential_privacy import compute_stdv

from metricdp_pytorch.dp_diagnostics import (
    add_diagnostics,
    clipping_diagnostics,
    signal_to_noise_diagnostics,
)


class LoggedGlobalDPServerSideFixedClipping(
    DifferentialPrivacyServerSideFixedClipping
):
    """Fixed-clipping global DP which records the values actually applied."""

    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_norm: float,
        num_sampled_clients: int,
    ) -> None:
        super().__init__(
            strategy=strategy,
            noise_multiplier=noise_multiplier,
            clipping_norm=clipping_norm,
            num_sampled_clients=num_sampled_clients,
        )
        self.current_noise_stdv = compute_stdv(
            noise_multiplier, clipping_norm, num_sampled_clients
        )
        self.current_round_diagnostics: dict[str, float | int] = {}

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        reply_list = list(replies)
        self.current_round_diagnostics = {}
        diagnostics = clipping_diagnostics(
            reply_list,
            current_arrays=self.current_arrays,
            clipping_norm=self.clipping_norm,
        )
        aggregated_arrays, aggregated_metrics = super().aggregate_train(
            server_round, reply_list
        )
        diagnostics.update(self.current_round_diagnostics)
        diagnostics["global-dp-noise-stdv"] = self.current_noise_stdv
        return aggregated_arrays, add_diagnostics(aggregated_metrics, diagnostics)

    def _add_noise_to_aggregated_arrays(
        self, aggregated_arrays: ArrayRecord
    ) -> ArrayRecord:
        self.current_round_diagnostics = signal_to_noise_diagnostics(
            aggregated_arrays,
            current_arrays=self.current_arrays,
            noise_stdv=self.current_noise_stdv,
        )
        return super()._add_noise_to_aggregated_arrays(aggregated_arrays)
