"""Residual clipping around the previous released update (WP6, D4 item 2).

Round ``t`` releases

    theta_{t+1} = theta_t + c_t + (1/n) sum_{i present} clip(u_i - c_t, rho_t) + N(0, sigma_t^2 I),
    sigma_t = noise_multiplier * rho_t / n,

with ``u_i`` client ``i``'s model update, ``n`` the public slot count (absent slots contribute a zero residual),
``c_t`` the previous released update (``c_1 = 0``; ``center="zero"`` keeps ``c_t = 0``) and ``rho_t`` either fixed or
the private quantile of the residual norms (Andrew et al. 2021: one bit ``1[||u_i - c_t|| <= rho_t]`` per client, a
Gaussian-noised count over the public slot count, a geometric update). With ``center="zero"`` and ``rho = C`` the
mechanism is plain clipping with a fixed denominator, the baseline at the same guarantee.

Privacy (proof: ``results/cia_frontier/residual_clipping/THEORY.md``): adding or removing one client changes the
residual sum by at most ``rho_t``, so every round is ``sqrt(1/z^2 + 1/sigma_b^2)``-GDP (no bit term with a fixed
``rho``), and ``T`` rounds compose to ``sqrt(T)`` times that.
"""

from __future__ import annotations

from collections.abc import Iterable
import math
from logging import INFO

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, Strategy

from metricdp_pytorch.dp_diagnostics import add_diagnostics, clipping_diagnostics
from metricdp_pytorch.private_metric_strategy import DEFAULT_LEARNING_RATE, quantile_update
from metricdp_pytorch.strategy_factory import UNIT_WEIGHT_KEY

CENTERS = ("previous", "zero")


def flat_norm(arrays) -> float:
    return float(math.sqrt(sum(float(np.square(a, dtype=np.float64).sum()) for a in arrays)))


def residual_round_account(noise_multiplier: float, bit_noise_std: float | None) -> float:
    """Per-round GDP parameter: ``1/z`` for the update, plus ``1/sigma_b`` for the quantile bits."""
    mu_update = 1.0 / noise_multiplier
    return mu_update if bit_noise_std is None else math.hypot(mu_update, 1.0 / bit_noise_std)


class ResidualClippingServerSide(DifferentialPrivacyServerSideFixedClipping):
    """Server-side residual clipping with a public slot denominator and equal client weights."""

    def __init__(
        self,
        strategy: Strategy,
        noise_multiplier: float,
        num_slots: int,
        *,
        center: str = "previous",
        rho: float | None = None,
        target_quantile: float | None = None,
        initial_rho: float | None = None,
        minimum_rho: float | None = None,
        maximum_rho: float | None = None,
        bit_noise_std: float | None = None,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        seed: int | None = None,
        arrayrecord_key: str = "arrays",
    ) -> None:
        adaptive = target_quantile is not None
        if adaptive == (rho is not None):
            raise ValueError("Give either a fixed rho or a target_quantile for a private rho.")
        if adaptive and None in (initial_rho, minimum_rho, maximum_rho, bit_noise_std):
            raise ValueError("A private rho needs initial_rho, minimum_rho, maximum_rho and bit_noise_std.")
        if adaptive and not (0 < minimum_rho <= initial_rho <= maximum_rho and 0 < target_quantile < 1
                             and bit_noise_std > 0 and learning_rate > 0):
            raise ValueError("Require 0 < minimum_rho <= initial_rho <= maximum_rho, 0 < target_quantile < 1, "
                             "bit_noise_std > 0 and learning_rate > 0.")
        if not adaptive and rho <= 0:
            raise ValueError("rho must be positive.")
        if center not in CENTERS:
            raise ValueError(f"center must be one of {CENTERS}.")
        if noise_multiplier <= 0 or num_slots < 1:
            raise ValueError("noise_multiplier must be positive and num_slots at least 1.")
        if getattr(strategy, "weighted_by_key", None) != UNIT_WEIGHT_KEY:
            raise ValueError(f"Residual clipping requires equal client weights (weighted_by_key={UNIT_WEIGHT_KEY!r}).")
        start = rho if rho is not None else initial_rho
        super().__init__(strategy=strategy, noise_multiplier=noise_multiplier, clipping_norm=start,
                         num_sampled_clients=num_slots)
        self.num_slots, self.center = num_slots, center
        self.rho = start
        self.adaptive = adaptive
        self.target_quantile, self.learning_rate = target_quantile, learning_rate
        self.minimum_rho, self.maximum_rho = minimum_rho, maximum_rho
        self.bit_noise_std = bit_noise_std if adaptive else None
        self.arrayrecord_key = arrayrecord_key
        self.rng = np.random.default_rng(seed)
        self.previous_update: list[np.ndarray] | None = None
        self.rounds_released = 0

    def __repr__(self) -> str:
        return "Residual clipping DP Strategy (Server-Side)"

    def summary(self) -> None:
        log(INFO, "\t├──> Residual clipping: center=%s, rho=%s (%s), z=%s, slots=%s", self.center, self.rho,
            "private quantile" if self.adaptive else "fixed", self.noise_multiplier, self.num_slots)
        self.strategy.summary()

    @property
    def mu_per_round(self) -> float:
        return residual_round_account(self.noise_multiplier, self.bit_noise_std)

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        reply_list = list(replies)
        if any(reply.has_error() for reply in reply_list) or not reply_list:
            return super().aggregate_train(server_round, reply_list)
        if len(reply_list) > self.num_slots:
            raise ValueError(f"{len(reply_list)} replies exceed the {self.num_slots} public slots.")
        keys = list(self.current_arrays.keys())
        current = [np.asarray(a, dtype=np.float64) for a in self.current_arrays.to_numpy_ndarrays()]
        center = ([np.zeros_like(a) for a in current] if self.center == "zero" or self.previous_update is None
                  else self.previous_update)
        rho = self.rho  # fixed by the public state before the round
        diagnostics = clipping_diagnostics(reply_list, current_arrays=self.current_arrays, clipping_norm=rho,
                                           arrayrecord_key=self.arrayrecord_key)
        total = [np.zeros_like(a) for a in current]
        residual_norms, bits = [], 0
        for reply in reply_list:
            record = reply.content.get(self.arrayrecord_key)
            if not isinstance(record, ArrayRecord):
                raise ValueError(f"Client reply is missing ArrayRecord {self.arrayrecord_key!r}.")
            model = [np.asarray(a, dtype=np.float64) for a in record.to_numpy_ndarrays()]
            residual = [m - c - k for m, c, k in zip(model, current, center, strict=True)]
            norm = flat_norm(residual)
            residual_norms.append(norm)
            bits += int(norm <= rho)
            scale = min(1.0, rho / norm) if norm > 0 else 1.0
            for slot, value in zip(total, residual, strict=True):
                slot += scale * value
        sigma = self.noise_multiplier * rho / self.num_slots
        update = [k + t / self.num_slots + self.rng.normal(0.0, sigma, size=t.shape)
                  for k, t in zip(center, total, strict=True)]
        released = [c + u for c, u in zip(current, update, strict=True)]
        dtypes = [np.asarray(a).dtype for a in self.current_arrays.to_numpy_ndarrays()]
        arrays = ArrayRecord({key: Array(np.asarray(value, dtype=dtype))
                              for key, value, dtype in zip(keys, released, dtypes, strict=True)})
        # Recompute the update from the stored (cast) arrays so the next centre is exactly the released update.
        self.previous_update = [np.asarray(value, dtype=np.float64) - c
                                for value, c in zip(arrays.to_numpy_ndarrays(), current, strict=True)]
        noisy_fraction = None
        if self.adaptive:
            noisy_fraction = (bits + float(self.rng.normal(0.0, self.bit_noise_std))) / self.num_slots
            self.rho = quantile_update(rho, noisy_fraction, target=self.target_quantile,
                                       learning_rate=self.learning_rate, minimum=self.minimum_rho,
                                       maximum=self.maximum_rho)
        self.rounds_released += 1
        _, metrics = self.strategy.aggregate_train(server_round, reply_list)  # client metrics only
        diagnostics.update({
            "residual-rho": rho,
            "residual-next-rho": self.rho,
            "residual-noise-stdv": sigma,
            "residual-noise-l2-norm-expected": sigma * math.sqrt(sum(a.size for a in current)),
            "residual-center-norm": flat_norm(center),
            "residual-released-update-norm": flat_norm(self.previous_update),
            "residual-norms": residual_norms,
            "residual-fraction-clipped": 1.0 - bits / len(reply_list),
            "residual-active-clients": len(reply_list),
            "residual-mu-per-round": self.mu_per_round,
            "residual-mu-total": math.sqrt(self.rounds_released) * self.mu_per_round,
        })
        if noisy_fraction is not None:
            diagnostics["residual-noisy-fraction"] = noisy_fraction
        log(INFO, "aggregate_train: residual clipping rho=%.4f, noise std %.6f, %d/%d clipped", rho, sigma,
            len(reply_list) - bits, len(reply_list))
        return arrays, add_diagnostics(metrics, diagnostics)
