"""Metric privacy with a privately released median client distance (WP3, D4 item 1).

The paper's metric privacy sets the round's noise from the maximum pairwise distance between
client models, a statistic computed from the participants that leaks (WP1, gate G1(a)) and has
no privacy accounting. This wrapper replaces it by a private estimate ``q_t`` of the **median**
client distance, tracked with the geometric quantile update of Andrew et al. (2021):

* each client ``i`` contributes one bit ``b_i = 1[s_i <= q_t]``, where ``s_i`` is the paper's
  mean layer-wise Euclidean distance between the client's returned model and the current
  global model;
* the server releases the noisy fraction ``(sum_i b_i + N(0, bit_noise_std^2)) / num_slots``
  over a public slot count, and updates ``q_{t+1} = clamp(q_t exp(-eta (fraction - gamma)))``;
* the round's update noise uses ``q_t`` (fixed before the round's data are seen):
  ``sigma_t = noise_multiplier * C / (num_slots * q_t)`` on the average, with ``q_t`` clamped to
  public bounds, so ``sigma_t >= sigma_min = noise_multiplier * C / (num_slots * q_max)``;
* the average uses the public slot count as denominator (equal weights), so adding or
  removing one client changes the clipped sum by at most ``C`` whatever the participant count.

Privacy (proof: ``research/cia_review_2026-10/theory/private_dispersion.md``): every round is
``sqrt(mu_update^2 + mu_bits^2)``-GDP with ``mu_update = q_max / noise_multiplier`` and
``mu_bits = 1 / bit_noise_std``; ``T`` rounds compose to ``sqrt(T)`` times that.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import math
from logging import INFO

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, Strategy
from flwr.supercore.differential_privacy import add_gaussian_noise_inplace, compute_stdv

from metricdp_pytorch.dp_diagnostics import add_diagnostics, clipping_diagnostics
from metricdp_pytorch.strategy_factory import UNIT_WEIGHT_KEY

# Andrew et al. (2021), "Differentially private learning with adaptive clipping": eta = 0.2.
DEFAULT_LEARNING_RATE = 0.2
DEFAULT_MEDIAN_BUDGET_SHARE = 0.1


def model_distance(model: Sequence[np.ndarray], reference: Sequence[np.ndarray]) -> float:
    """Mean over layers of the per-layer Euclidean distance (the paper's model distance)."""
    if len(model) != len(reference) or any(a.shape != b.shape for a, b in zip(model, reference, strict=True)):
        raise ValueError("Model and reference must have matching array shapes.")
    if not model:
        raise ValueError("Models must have at least one array.")
    return float(np.mean([np.linalg.norm((a - b).ravel()) for a, b in zip(model, reference, strict=True)]))


def quantile_update(quantile: float, noisy_fraction: float, *, target: float, learning_rate: float,
                    minimum: float, maximum: float) -> float:
    """Geometric private-quantile step, clamped to the public bounds."""
    return float(min(max(quantile * math.exp(-learning_rate * (noisy_fraction - target)), minimum), maximum))


def gaussian_delta(mu: float, epsilon: float) -> float:
    """``delta(epsilon)`` of a mu-GDP mechanism (Dong, Roth and Su 2022, Corollary 2.13)."""
    from statistics import NormalDist

    phi = NormalDist().cdf
    if mu == 0:
        return 0.0
    return phi(-epsilon / mu + mu / 2) - math.exp(epsilon) * phi(-epsilon / mu - mu / 2)


def gaussian_epsilon(mu: float, delta: float) -> float:
    """Smallest epsilon with ``delta(epsilon) <= delta`` for a mu-GDP mechanism (bisection)."""
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    if mu == 0 or gaussian_delta(mu, 0.0) <= delta:
        return 0.0
    low, high = 0.0, 1.0
    while gaussian_delta(mu, high) > delta:
        high *= 2
    for _ in range(200):
        middle = (low + high) / 2
        low, high = (middle, high) if gaussian_delta(mu, middle) > delta else (low, middle)
    return high


def bit_noise_for_share(noise_multiplier: float, maximum: float, share: float = DEFAULT_MEDIAN_BUDGET_SHARE) -> float:
    """Bit-count noise std that gives the median release ``share`` of each round's ``mu^2``."""
    if not 0 < share < 1:
        raise ValueError("share must be in (0, 1)")
    mu_update = maximum / noise_multiplier
    return math.sqrt((1 - share) / share) / mu_update


@dataclass(frozen=True)
class PrivacyAccount:
    """GDP accounting of the private-median mechanism (theory/private_dispersion.md, Theorem 1)."""

    mu_update_per_round: float
    mu_bits_per_round: float
    rounds: int

    @property
    def mu_per_round(self) -> float:
        return math.hypot(self.mu_update_per_round, self.mu_bits_per_round)

    @property
    def mu_total(self) -> float:
        return math.sqrt(self.rounds) * self.mu_per_round

    @property
    def median_share(self) -> float:
        """Share of the composed ``mu^2`` spent on the median release."""
        return self.mu_bits_per_round ** 2 / self.mu_per_round ** 2

    def epsilon(self, delta: float) -> float:
        return gaussian_epsilon(self.mu_total, delta)


def privacy_account(*, noise_multiplier: float, maximum: float, bit_noise_std: float, rounds: int) -> PrivacyAccount:
    if noise_multiplier <= 0 or maximum <= 0 or bit_noise_std <= 0 or rounds < 1:
        raise ValueError("noise_multiplier, maximum and bit_noise_std must be positive, rounds >= 1")
    return PrivacyAccount(maximum / noise_multiplier, 1.0 / bit_noise_std, rounds)


class PrivateMedianMetricPrivacyServerSideFixedClipping(DifferentialPrivacyServerSideFixedClipping):
    """Metric-aware server-side DP whose noise follows a privately tracked median client distance.

    ``noise_multiplier`` plays the paper's ``z``: the noise std on the average is
    ``noise_multiplier * clipping_norm / (num_slots * q_t)``. ``num_slots`` is the public
    denominator (the canonical client count); the wrapped strategy must weight clients
    equally (``weighted_by_key == UNIT_WEIGHT_KEY``).
    """

    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        clipping_norm: float,
        num_slots: int,
        *,
        initial_quantile: float,
        minimum_quantile: float,
        maximum_quantile: float,
        bit_noise_std: float | None = None,
        target_quantile: float = 0.5,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        seed: int | None = None,
        arrayrecord_key: str = "arrays",
    ) -> None:
        super().__init__(strategy=strategy, noise_multiplier=noise_multiplier, clipping_norm=clipping_norm,
                         num_sampled_clients=num_slots)
        if getattr(strategy, "weighted_by_key", None) != UNIT_WEIGHT_KEY:
            raise ValueError("The private-median mechanism requires equal client weights "
                             f"(weighted_by_key={UNIT_WEIGHT_KEY!r}).")
        if noise_multiplier <= 0:
            raise ValueError("noise_multiplier must be positive.")
        if not 0 < minimum_quantile <= initial_quantile <= maximum_quantile:
            raise ValueError("Require 0 < minimum_quantile <= initial_quantile <= maximum_quantile.")
        if not 0 < target_quantile < 1 or learning_rate <= 0:
            raise ValueError("Require 0 < target_quantile < 1 and learning_rate > 0.")
        self.bit_noise_std = (bit_noise_for_share(noise_multiplier, maximum_quantile)
                              if bit_noise_std is None else bit_noise_std)
        if self.bit_noise_std <= 0:
            raise ValueError("bit_noise_std must be positive.")
        self.num_slots = num_slots
        self.quantile = initial_quantile
        self.minimum_quantile, self.maximum_quantile = minimum_quantile, maximum_quantile
        self.target_quantile, self.learning_rate = target_quantile, learning_rate
        self.arrayrecord_key = arrayrecord_key
        self.rng = np.random.default_rng(seed)
        self.rounds_released = 0
        self._active_clients = 0
        self._round_quantile = initial_quantile
        self._noise_stdv: float | None = None

    def __repr__(self) -> str:
        return "Private-median metric-aware DP Strategy (Server-Side Fixed Clipping)"

    def summary(self) -> None:
        log(INFO, "\t├──> Private-median metric-aware DP settings:")
        log(INFO, "\t│\t├── Noise multiplier: %s, slots: %s", self.noise_multiplier, self.num_slots)
        log(INFO, "\t│\t├── Quantile %s in [%s, %s], bit noise std %s",
            self.quantile, self.minimum_quantile, self.maximum_quantile, self.bit_noise_std)
        log(INFO, "\t│\t└── Clipping norm: %s", self.clipping_norm)
        self.strategy.summary()

    def account(self, rounds: int | None = None) -> PrivacyAccount:
        return privacy_account(noise_multiplier=self.noise_multiplier, maximum=self.maximum_quantile,
                               bit_noise_std=self.bit_noise_std, rounds=rounds or max(self.rounds_released, 1))

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        reply_list = list(replies)
        if any(reply.has_error() for reply in reply_list):
            return super().aggregate_train(server_round, reply_list)
        if len(reply_list) > self.num_slots:
            raise ValueError(f"{len(reply_list)} replies exceed the {self.num_slots} public slots.")

        reference = self.current_arrays.to_numpy_ndarrays()
        distances = []
        for reply in reply_list:
            record = reply.content.get(self.arrayrecord_key)
            if not isinstance(record, ArrayRecord):
                raise ValueError(f"Client reply is missing ArrayRecord {self.arrayrecord_key!r}.")
            distances.append(model_distance(record.to_numpy_ndarrays(), reference))

        # Round t uses q_t, which depends only on earlier private releases.
        quantile = self._round_quantile = self.quantile
        bits = sum(distance <= quantile for distance in distances)
        noisy_fraction = (bits + float(self.rng.normal(0.0, self.bit_noise_std))) / self.num_slots
        diagnostics = clipping_diagnostics(reply_list, current_arrays=self.current_arrays,
                                           clipping_norm=self.clipping_norm, arrayrecord_key=self.arrayrecord_key)
        self._active_clients = len(reply_list)
        self._noise_stdv = None
        aggregated_arrays, aggregated_metrics = super().aggregate_train(server_round, reply_list)
        self.quantile = quantile_update(quantile, noisy_fraction, target=self.target_quantile,
                                        learning_rate=self.learning_rate, minimum=self.minimum_quantile,
                                        maximum=self.maximum_quantile)
        self.rounds_released += 1
        account = self.account()
        diagnostics.update({
            "private-median-quantile": quantile,
            "private-median-next-quantile": self.quantile,
            "private-median-noisy-fraction": noisy_fraction,
            "private-median-bit-noise-std": self.bit_noise_std,
            "private-median-mu-per-round": account.mu_per_round,
            "private-median-mu-total": account.mu_total,
            "private-median-budget-share": account.median_share,
            "private-median-active-clients": len(reply_list),
            # Server-side diagnostics only (not part of the release): the exact statistics.
            "private-median-true-fraction": bits / self.num_slots,
            "private-median-true-median-distance": float(np.median(distances)) if distances else 0.0,
        })
        if self._noise_stdv is not None:
            diagnostics["private-median-noise-stdv"] = self._noise_stdv
        return aggregated_arrays, add_diagnostics(aggregated_metrics, diagnostics)

    def _add_noise_to_aggregated_arrays(self, aggregated_arrays: ArrayRecord) -> ArrayRecord:
        """Rescale the equal-weight mean to the public slot denominator, then add the round's noise."""
        current = self.current_arrays.to_numpy_ndarrays()
        scale = self._active_clients / self.num_slots
        aggregated = [c + scale * (a - c) for a, c in zip(aggregated_arrays.to_numpy_ndarrays(), current, strict=True)]
        stdv = compute_stdv(self.noise_multiplier / self._round_quantile, self.clipping_norm, self.num_slots)
        add_gaussian_noise_inplace(aggregated, stdv)
        self._noise_stdv = stdv
        log(INFO, "aggregate_train: private-median noise with %.6f stdev added (q=%.6f)", stdv, self._round_quantile)
        return ArrayRecord({key: Array(np.asarray(value))
                            for key, value in zip(aggregated_arrays.keys(), aggregated, strict=True)})
