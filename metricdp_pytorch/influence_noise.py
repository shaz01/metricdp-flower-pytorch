"""Influence-directed Gaussian aggregate noise with matched expected energy.

EMPIRICAL MECHANISM, NO DP CLAIM. The covariance below depends on the private
client updates of the current round, so it is not a data-independent Gaussian
mechanism and no differential-privacy guarantee follows from it; the isotropic
full-rank part does not establish one either. ``d_i`` is a round-local
leave-one-out influence of the clipped aggregate, not the sensitivity of the
whole training trajectory to removing a client.

Per round, with clipped updates ``u_i`` (Flower's flat L2 clipping at ``C``)
and FedAvg sample weights ``w_i = n_i / sum_j n_j`` (all ``w_i < 1``):

    g     = sum_i w_i u_i
    d_i   = g - g_{-i} = (w_i / (1 - w_i)) (u_i - g)       (renormalised LOO)
    d_i  <- d_i * min(1, B / ||d_i||)                       (declared fixed cap)
    M     = sum_i d_i d_i^T,     trace(M) = sum_i ||d_i||^2
    Sigma = tau^2 [ (1 - f) I + f D M / trace(M) ]

``trace(Sigma) = tau^2 D`` for every ``f``, identical to the isotropic
comparator ``tau^2 I`` (the global-DP convention ``tau = z C / n``). The noise
therefore redistributes, rather than adds, expected squared norm. Sampling
never forms a D x D matrix:

    z = tau sqrt(1 - f) xi + tau sqrt(f D / trace(M)) sum_i eta_i d_i,
    xi ~ N(0, I_D), eta ~ N(0, I_n).

``f = 0`` is exactly isotropic. ``trace(M) == 0`` (all clipped updates equal)
falls back to the full isotropic ``tau^2 I``. ``xi`` and ``eta`` are always
drawn, in that order, from a generator keyed only by ``(seed, round)``, so the
isotropic and anisotropic arms share common random numbers per round.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from logging import WARNING

import numpy as np
from flwr.app import Array, ArrayRecord, Message, MetricRecord
from flwr.common import log
from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, Strategy
from flwr.serverapp.strategy.dp_fixed_clipping import validate_replies

from metricdp_pytorch.dp_diagnostics import add_diagnostics, clipping_diagnostics

RNG_DOMAIN_TAG = 0x1F1D  # separates this stream from any other seeded stream


def isotropic_stdv(noise_multiplier: float, clipping_norm: float, num_sampled_clients: int) -> float:
    """Exactly Flower's ``compute_stdv`` (global-DP convention)."""
    return float((noise_multiplier * clipping_norm) / num_sampled_clients)


def clip_update(update: Sequence[np.ndarray], clipping_norm: float) -> tuple[list[np.ndarray], float, float]:
    """Flower FlatClip (``min(1, C/||u||)``) with a zero-norm guard.

    Returns ``(clipped arrays, pre-clip norm, scale)``. The norm follows
    Flower's ``get_norm``. A zero update is left unchanged (scale 1) instead of
    raising ``ZeroDivisionError``.
    """
    norm = float(np.sqrt(sum(np.linalg.norm(a.flat) ** 2 for a in update)))
    scale = 1.0 if norm == 0.0 else min(1.0, clipping_norm / norm)
    return [a * scale for a in update], norm, scale


def sample_weights(num_examples: Sequence[float]) -> np.ndarray:
    counts = np.asarray(num_examples, dtype=np.float64)
    if counts.ndim != 1 or len(counts) < 2:
        raise ValueError("Influence noise needs at least two clients")
    if not np.all(np.isfinite(counts)) or np.any(counts <= 0):
        raise ValueError("Client example counts must be finite and positive")
    weights = counts / counts.sum()
    if not np.all(weights < 1.0):
        raise ValueError("Every aggregation weight must be < 1 for leave-one-out influence")
    return weights


def leave_one_out_influence(updates: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(g, d)`` with ``d[i] = g - g_{-i}`` for renormalised weights."""
    updates = np.asarray(updates, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    g = weights @ updates
    d = (weights / (1.0 - weights))[:, None] * (updates - g)
    if not np.all(np.isfinite(d)):
        raise FloatingPointError("Non-finite leave-one-out influence")
    return g, d


def cap_rows(vectors: np.ndarray, bound: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale each row to L2 norm <= ``bound``; return (capped, raw norms, capped flags)."""
    if not np.isfinite(bound) or bound <= 0:
        raise ValueError("Influence cap must be finite and positive")
    norms = np.linalg.norm(vectors, axis=1)
    scale = np.where(norms > bound, bound / np.where(norms > 0, norms, 1.0), 1.0)
    return vectors * scale[:, None], norms, norms > bound


@dataclass(frozen=True)
class NoiseDraw:
    noise: np.ndarray
    fallback_isotropic: bool
    trace: float
    directional_scale: float  # tau * sqrt(f D / trace), 0 when unused


def round_generator(seed: int, server_round: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([RNG_DOMAIN_TAG, int(seed), int(server_round)]))


def draw_noise(influence: np.ndarray, *, tau: float, fraction: float, rng: np.random.Generator) -> NoiseDraw:
    """Sample z ~ N(0, Sigma) using independent scalars times influence vectors."""
    if not 0.0 <= fraction <= 1.0 or not np.isfinite(fraction):
        raise ValueError("Orientation fraction must be in [0, 1]")
    if not np.isfinite(tau) or tau < 0:
        raise ValueError("tau must be finite and non-negative")
    n, dim = influence.shape
    xi = rng.standard_normal(dim)
    eta = rng.standard_normal(n)
    trace = float(np.sum(influence * influence))
    if fraction == 0.0:
        return NoiseDraw(tau * xi, False, trace, 0.0)
    if not (trace > 0.0 and np.isfinite(trace)):
        return NoiseDraw(tau * xi, True, trace, 0.0)
    directional = tau * np.sqrt(fraction * dim / trace)
    noise = tau * np.sqrt(1.0 - fraction) * xi + directional * (eta @ influence)
    return NoiseDraw(noise, False, trace, float(directional))


def covariance(influence: np.ndarray, *, tau: float, fraction: float) -> np.ndarray:
    """Dense Sigma for low-dimensional tests only."""
    n, dim = influence.shape
    trace = float(np.sum(influence * influence))
    if fraction == 0.0 or trace <= 0.0:
        return tau**2 * np.eye(dim)
    m = influence.T @ influence
    return tau**2 * ((1 - fraction) * np.eye(dim) + fraction * dim * m / trace)


def directional_stdv(influence: np.ndarray, *, tau: float, fraction: float, trace: float,
                     fallback: bool) -> np.ndarray:
    """Noise std along each unit direction d_i / ||d_i|| (0-norm rows -> isotropic)."""
    n, dim = influence.shape
    if fraction == 0.0 or fallback:
        return np.full(n, tau)
    gram = influence @ influence.T
    norms_sq = np.diag(gram)
    safe = np.where(norms_sq > 0, norms_sq, 1.0)
    quad = np.where(norms_sq > 0, np.sum(gram**2, axis=1) / safe, 0.0)
    return tau * np.sqrt((1 - fraction) + fraction * dim * quad / trace)


def _client_id(reply: Message) -> int:
    return int(next(iter(reply.content.metric_records.values()))["client-id"])


class InfluenceNoiseServerSideFixedClipping(DifferentialPrivacyServerSideFixedClipping):
    """Server-side fixed clipping + influence-directed matched-energy noise."""

    def __init__(self, strategy: Strategy, *, noise_multiplier: float, clipping_norm: float,
                 num_sampled_clients: int, fraction: float, influence_cap: float, seed: int) -> None:
        super().__init__(strategy, noise_multiplier, clipping_norm, num_sampled_clients)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("Orientation fraction must be in [0, 1]")
        if not np.isfinite(influence_cap) or influence_cap <= 0:
            raise ValueError("Influence cap must be finite and positive")
        if getattr(strategy, "weighted_by_key", None) != "num-examples":
            raise ValueError("Influence noise assumes FedAvg weighting by 'num-examples'")
        self.fraction = float(fraction)
        self.influence_cap = float(influence_cap)
        self.seed = int(seed)
        self.tau = isotropic_stdv(noise_multiplier, clipping_norm, num_sampled_clients)

    def __repr__(self) -> str:
        return "Influence-directed matched-energy noise (server-side fixed clipping; not DP)"

    def aggregate_train(self, server_round: int, replies: Iterable[Message]
                        ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        reply_list = list(replies)
        if not validate_replies(reply_list, self.num_sampled_clients):
            return None, None
        diagnostics = clipping_diagnostics(reply_list, current_arrays=self.current_arrays,
                                           clipping_norm=self.clipping_norm)
        reply_list.sort(key=_client_id)
        keys = list(self.current_arrays.keys())
        current = self.current_arrays.to_numpy_ndarrays()
        if any(not np.issubdtype(a.dtype, np.floating) for a in current):
            raise TypeError("Influence noise requires float parameters only (no integer buffers)")
        rows, counts, ids = [], [], []
        for reply in reply_list:
            if len(reply.content.array_records) != 1:
                raise ValueError("Expected exactly one ArrayRecord per reply")
            (name, record), = reply.content.array_records.items()
            if list(record.keys()) != keys:
                raise ValueError("Reply array keys differ from the global model")
            received = record.to_numpy_ndarrays()
            update = [np.subtract(x, y) for x, y in zip(received, current, strict=True)]
            clipped, _, _ = clip_update(update, self.clipping_norm)
            reply.content[name] = ArrayRecord(dict(zip(
                keys, (Array(np.asarray(y + u)) for y, u in zip(current, clipped, strict=True)),
                strict=True)))
            rows.append(np.concatenate([u.astype(np.float64).ravel() for u in clipped]))
            metrics = next(iter(reply.content.metric_records.values()))
            counts.append(float(metrics["num-examples"]))
            ids.append(int(metrics["client-id"]))
        aggregated, metrics = self.strategy.aggregate_train(server_round, reply_list)
        if aggregated is None:
            log(WARNING, "influence-noise: base strategy returned no arrays at round %d", server_round)
            return None, add_diagnostics(metrics, diagnostics)

        updates = np.stack(rows)
        weights = sample_weights(counts)
        _, influence = leave_one_out_influence(updates, weights)
        capped, raw_norms, cap_bound = cap_rows(influence, self.influence_cap)
        draw = draw_noise(capped, tau=self.tau, fraction=self.fraction,
                          rng=round_generator(self.seed, server_round))
        noisy, offset = [], 0
        signal_sq = 0.0
        for key, base, array in zip(keys, current, aggregated.to_numpy_ndarrays(), strict=True):
            piece = draw.noise[offset:offset + array.size].reshape(array.shape)
            offset += array.size
            signal_sq += float(np.sum(np.square(array - base, dtype=np.float64)))
            noisy.append(array + piece.astype(array.dtype))
        if offset != updates.shape[1]:
            raise ValueError("Aggregated arrays do not match the update dimension")
        dim = updates.shape[1]
        directional = directional_stdv(capped, tau=self.tau, fraction=self.fraction,
                                       trace=draw.trace, fallback=draw.fallback_isotropic)
        diagnostics.update({
            "influence-client-ids": ids,
            "influence-weights": weights.tolist(),
            "influence-clipped-update-norms": np.linalg.norm(updates, axis=1).tolist(),
            "influence-weighted-clipped-norms": (weights * np.linalg.norm(updates, axis=1)).tolist(),
            "influence-raw-norms": raw_norms.tolist(),
            "influence-capped-norms": np.linalg.norm(capped, axis=1).tolist(),
            "influence-cap-bound": [int(x) for x in cap_bound],
            "influence-directional-noise-stdv": directional.tolist(),
            "influence-trace": draw.trace,
            "influence-fallback-isotropic": int(draw.fallback_isotropic),
            "influence-fraction": self.fraction,
            "influence-cap": self.influence_cap,
            "influence-tau": self.tau,
            "influence-dimension": dim,
            "influence-expected-noise-sq-norm": self.tau**2 * dim,
            "influence-realized-noise-sq-norm": float(draw.noise @ draw.noise),
            "influence-directional-scale": draw.directional_scale,
            "dp-signal-update-norm": float(np.sqrt(signal_sq)),
            "dp-parameter-count": dim,
            "dp-expected-noise-l2-norm": float(self.tau * np.sqrt(dim)),
            "global-dp-noise-stdv": self.tau,
        })
        return ArrayRecord(dict(zip(keys, (Array(np.asarray(v)) for v in noisy), strict=True))), \
            add_diagnostics(metrics, diagnostics)
