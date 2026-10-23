"""Global DP with Poisson client sampling and a public denominator (CIA roadmap v2, B1).

Each round, every one of the ``num_clients`` active clients joins independently with probability ``q``. The release
is the current model plus ``sum_{j in S} clip(u_j, C) / (q N) + N(0, (z C / (q N))^2 I)``: the expected count ``q N`` is
the public denominator, so neither the noise nor the scaling depends on who was sampled. A round with no sampled
client releases noise only. Under add/remove-one-client adjacency this is the Poisson-subsampled Gaussian mechanism
with noise multiplier ``z`` (``metricdp_pytorch.accounting.SubsampledGaussian(z, q, T)``).

The sample is private randomness: it is drawn from its own generator, seeded by the run's noise seed when one is set
(``set_noise_seed``), and stays secret, which amplification requires.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord
from flwr.serverapp.strategy import Strategy

from metricdp_pytorch.dp_diagnostics import add_diagnostics
from metricdp_pytorch.globaldp_strategy import LoggedGlobalDPServerSideFixedClipping

SAMPLING_STREAM = 1  # the sampling generator is SeedSequence([seed, SAMPLING_STREAM]), apart from the noise stream


class PoissonSampledGlobalDP(LoggedGlobalDPServerSideFixedClipping):
    """Fixed-clipping global DP over a Poisson sample of the active clients, divided by ``q N``."""

    def __init__(self, strategy: Strategy, noise_multiplier: float, clipping_norm: float, num_clients: int,
                 sampling_rate: float, seed: int = 0) -> None:
        if not 0.0 < sampling_rate <= 1.0:
            raise ValueError("sampling_rate must be in (0, 1].")
        super().__init__(strategy=strategy, noise_multiplier=noise_multiplier, clipping_norm=clipping_norm,
                         num_sampled_clients=num_clients)
        self.num_clients = num_clients
        self.sampling_rate = sampling_rate
        self.expected_clients = sampling_rate * num_clients
        self.current_noise_stdv = noise_multiplier * clipping_norm / self.expected_clients
        self.reseed_sampling(seed)
        self.sampled_this_round: int | None = None

    def reseed_sampling(self, seed: int) -> None:
        self.sampling_rng = np.random.default_rng(np.random.SeedSequence([int(seed), SAMPLING_STREAM]))

    def configure_train(self, server_round, arrays, config, grid) -> Iterable[Message]:
        messages = list(super().configure_train(server_round, arrays, config, grid))
        if len(messages) != self.num_clients:
            raise RuntimeError(f"expected {self.num_clients} active clients to sample from, got {len(messages)}")
        keep = self.sampling_rng.random(len(messages)) < self.sampling_rate
        chosen = [message for message, kept in zip(messages, keep, strict=True) if kept]
        self.sampled_this_round = len(chosen)
        return chosen

    def aggregate_train(self, server_round: int,
                        replies: Iterable[Message]) -> tuple[ArrayRecord | None, MetricRecord | None]:
        reply_list = list(replies)
        self._round_replies = len(reply_list)
        diagnostics = {"subsampled-dp-sampled-clients": len(reply_list),
                       "subsampled-dp-sampling-rate": self.sampling_rate,
                       "subsampled-dp-denominator": self.expected_clients}
        if not reply_list:  # nobody sampled: release the current model plus noise
            released = self._add_noise_to_aggregated_arrays(self.current_arrays)
            diagnostics.update(self.current_round_diagnostics)
            diagnostics.update(self.noise_round_diagnostics)
            diagnostics["global-dp-noise-stdv"] = self.current_noise_stdv
            return released, add_diagnostics(None, diagnostics)
        arrays, metrics = super().aggregate_train(server_round, reply_list)
        return arrays, add_diagnostics(metrics, diagnostics)

    def _add_noise_to_aggregated_arrays(self, aggregated_arrays: ArrayRecord) -> ArrayRecord:
        """Rescale the equal-weight mean of the sampled clients to the public denominator ``q N``, then add noise."""
        current = self.current_arrays.to_numpy_ndarrays()
        scale = self._round_replies / self.expected_clients
        scaled = [c + scale * (a - c) for a, c in zip(aggregated_arrays.to_numpy_ndarrays(), current, strict=True)]
        record = ArrayRecord({key: Array(np.asarray(value))
                              for key, value in zip(aggregated_arrays.keys(), scaled, strict=True)})
        return super()._add_noise_to_aggregated_arrays(record)
