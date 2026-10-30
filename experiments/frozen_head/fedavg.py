"""Client-level DP-FedAvg on a linear softmax head over fixed features (in process, float64 NumPy).

One round from the released head ``theta``:

1. every active client runs ``local_epochs`` epochs of minibatch SGD on its mean cross-entropy,
   with batch order from ``default_rng([client_seed, client, round])``;
2. its update ``theta_local - theta`` is clipped to L2 norm ``clip``;
3. the server releases ``theta + server_lr * (sum_clipped / num_slots + xi)`` with
   ``xi ~ N(0, (noise_multiplier * clip / num_slots)^2 I)`` drawn from ``default_rng(noise_seed)``.

The denominator is the public slot count whatever the active set, and the noise std does not
depend on who takes part (roadmap v2 Section 1, decision 4). Adding or removing one client changes
the clipped sum by at most ``clip``, so one round is ``(1 / noise_multiplier)``-GDP and ``T`` rounds
are ``sqrt(T) / noise_multiplier``-GDP (``metricdp_pytorch.accounting.gdp_mu``).

Two options of roadmap v3 P5 (``results/cia_v3/formal_cia/PROTOCOL.md`` Section 4); their defaults leave the path above
unchanged:

* ``calibration="metric"``: the paper's distance calibration (``metricdp_pytorch.metricdp_strategy``), noise std
  ``(noise_multiplier / d_t) * clip / num_slots`` with ``d_t`` the maximum pairwise mean layer-wise (``W``, ``b``)
  distance between the active clients' local models ``theta + u_i`` (unclipped). No formal guarantee.
* ``sampling_rate = q < 1``: Poisson client sampling from ``default_rng(sampling_seed)``; the sampled clipped sum is
  divided by ``q * num_slots`` and the noise std is ``noise_multiplier * clip / (q * num_slots)``, so the noise does
  not depend on who was sampled (``metricdp_pytorch.subsampled_dp_strategy``; ``accounting.SubsampledGaussian``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HeadConfig:
    rounds: int
    local_epochs: int
    local_lr: float
    batch_size: int
    clip: float
    server_lr: float
    noise_multiplier: float  # 0 = no noise
    num_slots: int

    def as_dict(self) -> dict:
        return asdict(self)


class Head:
    """Parameter vector ``[W.ravel(), b]`` of a ``(classes, dim)`` softmax head."""

    def __init__(self, classes: int, dim: int) -> None:
        self.classes, self.dim = classes, dim
        self.size = classes * dim + classes

    def unpack(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return theta[: self.classes * self.dim].reshape(self.classes, self.dim), theta[self.classes * self.dim:]

    def pack(self, weights: np.ndarray, bias: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(weights, dtype=float).ravel(), np.asarray(bias, dtype=float)])

    def logits(self, theta: np.ndarray, features: np.ndarray) -> np.ndarray:
        weights, bias = self.unpack(theta)
        return features @ weights.T + bias

    def predict(self, theta: np.ndarray, features: np.ndarray) -> np.ndarray:
        return np.argmax(self.logits(theta, features), axis=1)

    def accuracy(self, theta: np.ndarray, features: np.ndarray, labels: np.ndarray) -> float:
        return float(np.mean(self.predict(theta, features) == labels))

    def loss_and_grad(self, theta: np.ndarray, features: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
        """Mean cross-entropy and its gradient."""
        logits = self.logits(theta, features)
        logits -= logits.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(logits).sum(axis=1))
        loss = float(np.mean(log_norm - logits[np.arange(len(labels)), labels]))
        probs = np.exp(logits - log_norm[:, None])
        probs[np.arange(len(labels)), labels] -= 1.0
        probs /= len(labels)
        return loss, self.pack(probs.T @ features, probs.sum(axis=0))


def local_update(head: Head, theta: np.ndarray, features: np.ndarray, labels: np.ndarray, config: HeadConfig,
                 rng: np.random.Generator) -> np.ndarray:
    """Unclipped update ``theta_local - theta`` after ``local_epochs`` of minibatch SGD."""
    local = theta.copy()
    for _ in range(config.local_epochs):
        order = rng.permutation(len(labels))
        for start in range(0, len(order), config.batch_size):
            batch = order[start:start + config.batch_size]
            _, grad = head.loss_and_grad(local, features[batch], labels[batch])
            local -= config.local_lr * grad
    return local - theta


def clip_update(update: np.ndarray, clip: float) -> np.ndarray:
    norm = float(np.linalg.norm(update))
    return update if norm <= clip else update * (clip / norm)


def max_pairwise_layer_distance(head: Head, models: np.ndarray) -> float:
    """Maximum over pairs of the mean of the per-layer (``W``, ``b``) Euclidean distances, as
    ``metricdp_strategy.pairwise_model_distances``; ``models`` is ``(clients, size)``."""
    split = head.classes * head.dim
    layers = []
    for block in (models[:, :split], models[:, split:]):
        squared = np.einsum("ij,ij->i", block, block)
        gram = block @ block.T
        layers.append(np.sqrt(np.maximum(squared[:, None] + squared[None, :] - 2.0 * gram, 0.0)))
    mean = (layers[0] + layers[1]) / 2.0
    return float(mean[np.triu_indices(len(models), k=1)].max()) if len(models) > 1 else float("nan")


def run(head: Head, init: np.ndarray, clients: Sequence[tuple[np.ndarray, np.ndarray]], config: HeadConfig, *,
        client_seed: int, noise_seed: int, active: Sequence[int] | None = None,
        evaluate: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        keep_updates: bool = False, record_thetas: bool = False, calibration: str = "fixed",
        sampling_rate: float = 1.0, sampling_seed: int | None = None) -> dict:
    """Run ``config.rounds`` rounds; ``clients[i] = (features, labels)`` of client ``i``'s train set.

    ``active`` lists the participating clients (default all); the denominator stays ``num_slots``.
    ``evaluate`` maps names to ``(features, labels)`` scored after every round. ``keep_updates``
    stores each round's clipped updates (``(rounds, len(active), size)``) and round-start heads
    (``(rounds, size)``) for the crowd bound. ``record_thetas`` stores the released head after every
    round (``(rounds, size)``) for attack scoring. ``calibration`` and ``sampling_rate`` are P5's options (module
    docstring); with either set, the output also has per-round ``noise_std_by_round``, and ``distance_by_round``
    (metric) or ``sampled_by_round`` (the sampled client ids, sampling).
    """
    if config.num_slots < 1 or config.clip <= 0 or config.noise_multiplier < 0:
        raise ValueError("Require num_slots >= 1, clip > 0 and noise_multiplier >= 0.")
    active = list(range(len(clients))) if active is None else list(active)
    if len(active) > config.num_slots:
        raise ValueError("More active clients than public slots.")
    if calibration not in ("fixed", "metric") or not 0 < sampling_rate <= 1:
        raise ValueError("calibration must be 'fixed' or 'metric' and sampling_rate in (0, 1].")
    if calibration == "metric" and sampling_rate < 1:
        raise ValueError("Metric calibration is defined for full participation only.")
    subsampled = sampling_rate < 1
    if subsampled and sampling_seed is None:
        raise ValueError("Client sampling needs its own sampling_seed (private randomness).")
    noise_rng = np.random.default_rng(noise_seed)
    sampling_rng = np.random.default_rng(sampling_seed) if subsampled else None
    denominator = sampling_rate * config.num_slots if subsampled else config.num_slots
    noise_std = config.noise_multiplier * config.clip / denominator
    theta = np.asarray(init, dtype=float).copy()
    history: dict[str, list] = {name: [] for name in (evaluate or {})}
    norms, clipped_fraction, updates, starts, released = [], [], [], [], []
    noise_by_round, distance_by_round, sampled_by_round = [], [], []
    distance = None
    for round_index in range(config.rounds):
        theta_start = theta.copy()
        total = np.zeros(head.size)
        round_norms, round_updates, local_models = [], [], []
        participants = active
        if subsampled:
            participants = [c for c, s in zip(active, sampling_rng.random(len(active)) < sampling_rate) if s]
            sampled_by_round.append(participants)
        for client in participants:
            features, labels = clients[client]
            rng = np.random.default_rng([client_seed, client, round_index])
            update = local_update(head, theta, features, labels, config, rng)
            round_norms.append(float(np.linalg.norm(update)))
            if calibration == "metric":
                local_models.append(theta + update)
            clipped = clip_update(update, config.clip)
            total += clipped
            if keep_updates:
                round_updates.append(clipped)
        round_std = noise_std
        if calibration == "metric":
            raw = max_pairwise_layer_distance(head, np.stack(local_models))
            if np.isfinite(raw) and raw > 0:
                distance = raw
            elif distance is None:
                distance = 1.0  # the strategy's fallback when no valid distance was seen yet
            distance_by_round.append(raw)
            round_std = noise_std / distance
        noise = noise_rng.normal(0.0, round_std, head.size) if round_std > 0 else np.zeros(head.size)
        noise_by_round.append(round_std)
        theta = theta + config.server_lr * (total / denominator + noise)
        if record_thetas:
            released.append(theta.copy())
        norms.append(float(np.median(round_norms)) if round_norms else float("nan"))
        clipped_fraction.append(float(np.mean(np.asarray(round_norms) > config.clip)) if round_norms else float("nan"))
        if keep_updates:
            updates.append(np.stack(round_updates))
            starts.append(theta_start)
        for name, (features, labels) in (evaluate or {}).items():
            history[name].append(head.accuracy(theta, features, labels))
    out = {"theta": theta, "accuracy": history, "median_update_norm": norms, "clipped_fraction": clipped_fraction,
           "noise_std": noise_std}
    if calibration == "metric" or subsampled:
        out["noise_std_by_round"] = noise_by_round
    if calibration == "metric":
        out["distance_by_round"] = distance_by_round
    if subsampled:
        out["sampled_by_round"] = sampled_by_round
    if record_thetas:
        out["thetas"] = np.stack(released)
    if keep_updates:
        out["updates"] = np.stack(updates)
        out["round_start_theta"] = np.stack(starts)
    return out
