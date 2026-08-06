"""Global-DP Flower wrapper with persisted per-round diagnostics."""

from __future__ import annotations

from collections.abc import Iterable
from logging import WARNING

from flwr.app import ArrayRecord, Message, MetricRecord
from flwr.common import log
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
        try:
            aggregated_arrays, aggregated_metrics = super().aggregate_train(
                server_round, reply_list
            )
        except ZeroDivisionError:
            # Flower's own DifferentialPrivacyServerSideFixedClipping.aggregate_train
            # (flwr.supercore.differential_privacy.clip_inputs_inplace) divides by a
            # client update's L2 norm to compute its clipping scale, with no guard
            # for a zero-norm update. metricdp_strategy.py's
            # MetricPrivacyServerSideFixedClipping already guards this same call for
            # its own route to the crash (calibrated noise blowing up when the
            # pairwise-distance denominator collapses); this strategy had no
            # equivalent guard, because the crash had only been observed via that
            # metric-privacy-specific route. It reaches here too, by a different
            # path: sustained high global-dp noise (--noise-multiplier gtrsim 0.25),
            # compounded round-over-round into the base model every client trains
            # from, can push local training into a state where a client's fresh
            # update is numerically indistinguishable from the current global model
            # -- observed live sweeping noise_multiplier up to 1.0
            # (results/noise_by_clients/sweep_progress.log). Returning None for
            # arrays keeps Strategy.start() on the previous round's model (see
            # flwr's Strategy.start(): "if agg_arrays is not None: arrays =
            # agg_arrays"), so a run survives one collapsed round instead of
            # crashing and losing every prior round's history.
            log(
                WARNING,
                "aggregate_train: round %d client updates collapsed to a "
                "zero-norm state -- Flower's clipping code can't handle a "
                "zero-magnitude update; skipping aggregation this round and "
                "keeping the previous round's model",
                server_round,
            )
            diagnostics["global-dp-aggregation-collapsed"] = 1.0
            return None, add_diagnostics(None, diagnostics)

        diagnostics["global-dp-aggregation-collapsed"] = 0.0
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
